//
// Created by robert-denomme on 8/27/24.
//
#ifndef DESCENT_IMPL_HPP
#define DESCENT_IMPL_HPP


namespace iceSYCL
{


template<class TInterpolationScheme>
template<typename ConstitutiveModel>
void Engine<TInterpolationScheme>::step_frame_implicit(const ConstitutiveModel Psi,
                                                       const size_t num_steps_per_frame,
                                                       const size_t num_descent_steps,
                                                       const double mu_velocity_damping,
                                                       const double gravity
)
{
    sycl::queue q{};
    auto q_policy = dpl::execution::make_device_policy(q);
    const scalar_t dt = 1.0 / (50 * num_steps_per_frame);
    //dt = 1/N => mu_step^N = mu_step^(1/dt) = mu
    //mu_step = mu^(dt)
    const scalar_t mu_step = std::pow(mu_velocity_damping, dt);


    for (size_t step = 0; step < num_steps_per_frame; ++step)
    {
        dpl::copy(q_policy, dpl::begin(particle_data.positions), dpl::end(particle_data.positions),
                  dpl::begin(particle_data.positions_prev));
        dpl::copy(q_policy, dpl::begin(particle_data.velocities), dpl::end(particle_data.velocities),
                  dpl::begin(particle_data.velocities_prev));
        dpl::copy(q_policy, dpl::begin(particle_data.deformation_gradients),
                  dpl::end(particle_data.deformation_gradients), dpl::begin(particle_data.deformation_gradients_prev));

        pgi_manager.update_particle_locations(q, particle_data.positions, interpolator);

        compute_node_inertial_positions(q, dt);
        update_particle_deformation_gradients_implicit(q, dt);
        for(size_t descent_step = 0; descent_step < num_descent_steps; ++descent_step)
        {
            compute_descent_gradient(q, Psi, dt, gravity, node_data.predicted_positions, descent_data.gradient);

            set_descent_direction(q);

            compute_directional_hessian(q, Psi, dt, gravity);

            initial_step(q);

            update_particle_deformation_gradients_implicit(q, dt);
        }

        compute_node_velocities_implicit(q, dt);
        transfer_velocity_nodes_to_particles_APIC(q);

        apply_particle_velocities_with_aether_damping(q, dt, mu_step);


    }

}

template<class TInterpolationScheme>
void Engine<TInterpolationScheme>::compute_node_inertial_positions(sycl::queue &q, const scalar_t dt)
{
    transer_mass_particles_to_nodes(q);
    transfer_momentum_particles_to_nodes_APIC(q);
    compute_node_velocities(q);

    auto interaction_access = pgi_manager.kernel_accessor;
    auto n = interpolator;
    q.submit([&](sycl::handler &h)
     {
         sycl::accessor positions_acc(node_data.predicted_positions, h);
         sycl::accessor inertial_positions_acc(node_data.inertial_positions, h);
         sycl::accessor velocities_acc(node_data.velocities, h);
         interaction_access.give_kernel_access(h);

         h.parallel_for(node_data.max_node_count, [=](sycl::id<1> idx)
         {
             const size_t node_count = interaction_access.node_count();
             const size_t node_id = idx[0];
             if (node_id >= node_count)
                 return;

             const NodeIndex_t i = interaction_access.get_node_index(node_id);
             const Coordinate_t x_i = n.position(i);

             const Coordinate_t inertial_position = x_i + dt * velocities_acc[node_id];
             positions_acc[node_id] = inertial_position;
             inertial_positions_acc[node_id] = inertial_position;
         });
     });
}

template<class TInterpolationScheme>
void Engine<TInterpolationScheme>::update_particle_deformation_gradients_implicit(sycl::queue &q, scalar_t dt)
{
    auto n = interpolator;
    auto interaction_access = pgi_manager.kernel_accessor;
    q.submit([&](sycl::handler &h)
     {
         sycl::accessor del_v_del_x_acc(particle_data.del_v_del_x, h);
         sycl::accessor particle_positions_acc(particle_data.positions, h);
         sycl::accessor node_predicted_positions_acc(node_data.predicted_positions, h);

         interaction_access.give_kernel_access(h);

         h.parallel_for(particle_count, [=](sycl::id<1> idx)
         {
             size_t pid = idx[0];
             Coordinate_t x_p = particle_positions_acc[pid];

             CoordinateMatrix_t del_v_del_x_p = CoordinateMatrix_t::Identity();
             for (auto interaction_it = interaction_access.particle_interactions_begin(pid);
                  interaction_it != interaction_access.particle_interactions_end(pid);
                  ++interaction_it)
             {
                 ParticleNodeInteraction<typename TInterpolationScheme::CoordinateConfiguration> interaction = *interaction_it;
                 size_t node_id = interaction.node_id;
                 NodeIndex_t node_index = interaction.node_index;
                 Coordinate_t x_i = n.position(node_index);

                 Coordinate_t x_i_predicted = node_predicted_positions_acc[node_id];

                 Coordinate_t grad_n_i = n.gradient(node_index, x_p);

                 del_v_del_x_p += (x_i_predicted - x_i) * grad_n_i.transpose();
             }
             del_v_del_x_acc[pid] = del_v_del_x_p;
         });
     });

    q.submit([&](sycl::handler &h)
     {
         sycl::accessor del_v_del_x_acc(particle_data.del_v_del_x, h);
         sycl::accessor particle_positions_acc(particle_data.positions, h);
         sycl::accessor deformation_gradients_acc(particle_data.deformation_gradients, h);
         sycl::accessor deformation_gradients_prev_acc(particle_data.deformation_gradients_prev, h);


         h.parallel_for(particle_count, [=](sycl::id<1> idx)
         {
             size_t pid = idx[0];
             Coordinate_t x_p = particle_positions_acc[pid];

             deformation_gradients_acc[pid] = del_v_del_x_acc[pid] * deformation_gradients_prev_acc[pid];
         });
     });


}

template<class TInterpolationScheme>
template<typename ConstitutiveModel>
void Engine<TInterpolationScheme>::compute_descent_gradient(
        sycl::queue &q,
        const ConstitutiveModel Psi,
        scalar_t dt,
        const double gravity,
        sycl::buffer<Coordinate_t> &node_positions,
        sycl::buffer<Coordinate_t> &gradient_destination
)
{
    auto interaction_access = pgi_manager.kernel_accessor;
    auto n = interpolator;
    //MPM forces
    q.submit([&](sycl::handler &h)
             {
                 sycl::accessor particle_mass_acc(particle_data.masses, h);
                 sycl::accessor particle_positions_acc(particle_data.positions, h);
                 sycl::accessor descent_gradient_acc(descent_data.gradient, h);
                 sycl::accessor deformation_gradient_acc(particle_data.deformation_gradients, h);
                 sycl::accessor deformation_gradient_prev_acc(particle_data.deformation_gradients_prev, h);
                 sycl::accessor rest_volume_acc(particle_data.rest_volumes, h);


                 interaction_access.give_kernel_access(h);

                 h.parallel_for(node_data.max_node_count, [=](sycl::id<1> idx)
                 {
                     const size_t node_count = interaction_access.node_count();
                     const size_t node_id = idx[0];
                     if (node_id >= node_count)
                         return;

                     Coordinate_t grad_i = Coordinate_t::Zero();
                     for (auto interaction_it = interaction_access.node_interactions_begin(node_id);
                          interaction_it != interaction_access.node_interactions_end(node_id);
                          ++interaction_it)
                     {
                         ParticleNodeInteraction<typename TInterpolationScheme::CoordinateConfiguration> interaction = *interaction_it;
                         size_t pid = interaction.particle_id;
                         scalar_t mass_p = particle_mass_acc[pid];
                         Coordinate_t x_p = particle_positions_acc[pid];
                         scalar_t V_p = rest_volume_acc[pid];
                         CoordinateMatrix_t F_prev = deformation_gradient_prev_acc[pid];
                         CoordinateMatrix_t F = deformation_gradient_acc[pid];
                         CoordinateMatrix_t PK = Psi.PK(F);

                         NodeIndex_t node_index = interaction.node_index;

                         grad_i += V_p * PK * F_prev.transpose() * n.gradient(node_index, x_p);

                     }

                     descent_gradient_acc[node_id] = grad_i;
                 });
             });

    //inertia + walls + gravity
    q.submit([&](sycl::handler &h)
             {
                 sycl::accessor node_mass_acc(node_data.masses, h);
                 sycl::accessor node_predicted_positions_acc(node_positions, h);
                 sycl::accessor node_inertial_positions_acc(node_data.inertial_positions, h);
                 sycl::accessor descent_gradient_acc(gradient_destination, h);
                 sycl::accessor walls_acc(collision_walls, h);


                 interaction_access.give_kernel_access(h);

                 h.parallel_for(node_data.max_node_count, [=](sycl::id<1> idx)
                 {
                     const size_t node_count = interaction_access.node_count();
                     const size_t node_id = idx[0];
                     if (node_id >= node_count)
                         return;

                     Coordinate_t gravity_vec = Coordinate_t(0.0, gravity);
                     NodeIndex_t node_index = interaction_access.get_node_index(node_id);
                     scalar_t mass_i = node_mass_acc[node_id];
                     Coordinate_t inertial_position = node_inertial_positions_acc[node_id];
                     Coordinate_t node_predicted_position = node_predicted_positions_acc[node_id];

                     descent_gradient_acc[node_id] += mass_i * gravity_vec;//.dot(node_predicted_position);
                     descent_gradient_acc[node_id] +=
                             mass_i / (dt * dt) * (node_predicted_position - inertial_position);

                     for (ElasticCollisionWall<CoordinateConfiguration> &wall: walls_acc)
                     {
                         descent_gradient_acc[node_id] += wall.gradient(node_predicted_position);
                     }
                 });
             });
}

template<class TInterpolationScheme>
void Engine<TInterpolationScheme>::set_descent_direction(
        sycl::queue &q
)
{
    auto interaction_access = pgi_manager.kernel_accessor;
    auto n = interpolator;

    q.submit([&](sycl::handler &h)
             {
                 sycl::accessor descent_gradient_acc(descent_data.gradient, h);
                 sycl::accessor descent_direction_acc(descent_data.descent_direction, h);

                 interaction_access.give_kernel_access(h);

                 h.parallel_for(node_data.max_node_count, [=](sycl::id<1> idx)
                 {
                     const size_t node_count = interaction_access.node_count();
                     const size_t node_id = idx[0];
                     if (node_id >= node_count)
                         return;
                     descent_direction_acc[node_id] = -descent_gradient_acc[node_id];
                 });
             });
}

template<class TInterpolationScheme>
template<typename ConstitutiveModel>
void Engine<TInterpolationScheme>::compute_directional_hessian(
        sycl::queue &q,
        const ConstitutiveModel Psi,
        scalar_t dt,
        const double gravity
)
{
    auto interaction_access = pgi_manager.kernel_accessor;
    auto n = interpolator;
    scalar_t epsilon = 1.0E-6;

    q.submit([&](sycl::handler &h)
     {
         sycl::accessor node_positions_acc(node_data.predicted_positions, h);
         sycl::accessor node_positions_plus_acc(descent_data.node_positions_plus, h);
         sycl::accessor descent_direction_acc(descent_data.descent_direction, h);

         interaction_access.give_kernel_access(h);

         h.parallel_for(node_data.max_node_count, [=](sycl::id<1> idx)
         {
             const size_t node_count = interaction_access.node_count();
             const size_t node_id = idx[0];
             if (node_id >= node_count)
                 return;
             node_positions_plus_acc[node_id] =
                     node_positions_acc[node_id] + epsilon * descent_direction_acc[node_id];
         });
     });

    compute_descent_gradient(q, Psi, dt, gravity, descent_data.node_positions_plus, descent_data.gradient_plus);

    initial_vec_dot(q, node_data.max_node_count, pgi_manager.node_count, descent_data.gradient, descent_data.descent_direction, descent_data.descent_direction_dot_grad);
    initial_vec_dot(q, node_data.max_node_count, pgi_manager.node_count, descent_data.gradient_plus, descent_data.descent_direction, descent_data.descent_direction_dot_grad_plus);

    q.submit([&](sycl::handler& h)
     {
        sycl::accessor directional_hessian_acc(descent_data.directional_hessian, h);
        sycl::accessor descent_direction_dot_grad_acc(descent_data.descent_direction_dot_grad, h);
        sycl::accessor descent_direction_dot_grad_plus_acc(descent_data.descent_direction_dot_grad_plus, h);

        h.single_task([=]()
          {
              directional_hessian_acc[0] = (descent_direction_dot_grad_plus_acc[0] - descent_direction_dot_grad_acc[0]) / epsilon;
          });
     });
}

template<class TInterpolationScheme>
void Engine<TInterpolationScheme>::initial_step(
        sycl::queue &q)
{

    auto interaction_access = pgi_manager.kernel_accessor;

    q.submit([&](sycl::handler& h)
     {
         sycl::accessor directional_hessian_acc(descent_data.directional_hessian, h);
         sycl::accessor descent_direction_dot_grad_acc(descent_data.descent_direction_dot_grad, h);
         sycl::accessor alpha_step_acc(descent_data.alpha_step, h);

         h.single_task([=]()
           {
               alpha_step_acc[0] = - descent_direction_dot_grad_acc[0] / directional_hessian_acc[0];
           });
     });

    q.submit([&](sycl::handler &h)
     {
         sycl::accessor node_positions_acc(node_data.predicted_positions, h);
         sycl::accessor alpha_step_acc(descent_data.alpha_step, h);
         sycl::accessor descent_direction_acc(descent_data.descent_direction, h);

         interaction_access.give_kernel_access(h);

         h.parallel_for(node_data.max_node_count, [=](sycl::id<1> idx)
         {
             const size_t node_count = interaction_access.node_count();
             const size_t node_id = idx[0];
             if (node_id >= node_count)
                 return;
             node_positions_acc[node_id] =
                     node_positions_acc[node_id] + alpha_step_acc[0] * descent_direction_acc[node_id];
         });
     });

}

//template<class TInterpolationScheme>
//void Engine<TInterpolationScheme>::compute_particle_velocities(sycl::queue &q, const scalar_t dt)
//{
//    auto n = interpolator;
//    auto interaction_access = pgi_manager.kernel_accessor;
//    q.submit([&](sycl::handler &h)
//             {
//                 sycl::accessor particle_positions_acc(particle_data.positions, h);
//                 sycl::accessor particle_positions_prev_acc(particle_data.positions_prev, h);
//                 sycl::accessor particle_velocities_acc(node_data.velocities, h);
//
//                 interaction_access.give_kernel_access(h);
//
//                 h.parallel_for(particle_count, [=](sycl::id<1> idx)
//                 {
//                     size_t pid = idx[0];
//                     Coordinate_t x_p = particle_positions_acc[pid];
//                     Coordinate_t x_p_prev = particle_positions_prev_acc[pid];
//
//                     particle_velocities_acc[pid] = 1.0 / dt * (x_p - x_p_prev);
//                 });
//             });
//
//}

template<class TInterpolationScheme>
void Engine<TInterpolationScheme>::compute_node_velocities_implicit(sycl::queue& q, const scalar_t dt)
{
    auto interaction_access = pgi_manager.kernel_accessor;
    auto n = interpolator;
    //nodes
    q.submit([&](sycl::handler& h)
     {
         sycl::accessor node_velocity_acc(node_data.velocities, h);
         sycl::accessor node_position_acc(node_data.predicted_positions, h);

         interaction_access.give_kernel_access(h);

         h.parallel_for(node_data.max_node_count,[=](sycl::id<1> idx)
         {
             const size_t node_count = interaction_access.node_count();
             const size_t node_id = idx[0];
             if(node_id >= node_count)
                 return;


             const NodeIndex_t i = interaction_access.get_node_index(node_id);
             const Coordinate_t x_i = n.position(i);

             node_velocity_acc[node_id] = 1.0 / dt * (node_position_acc[node_id] - x_i);
         });
     });
}

}

#endif //DESCENT_IMPL_HPP