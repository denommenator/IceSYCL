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

    auto interaction_access = pgi_manager.kernel_accessor;

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

        q.submit([&](sycl::handler& h){
            sycl::accessor gradients_acc(descent_data.gradient, h);
            sycl::accessor descent_direction_acc(descent_data.descent_direction, h);
            sycl::accessor descent_step_reset_counter_acc(descent_data.descent_step_reset_counter, h);

            interaction_access.give_kernel_access(h);

            h.parallel_for(node_data.max_node_count, [=](sycl::id<1> idx)
            {
                const size_t node_count = interaction_access.node_count();
                const size_t node_id = idx[0];
                if (node_id >= node_count)
                    return;
                gradients_acc[node_id] = Coordinate_t::Zero();
                descent_direction_acc[node_id] = Coordinate_t::Zero();
                if(node_id == 0)
                {
                    descent_step_reset_counter_acc[0] = 0;
                }
            });
        });

        for(size_t descent_step = 0; descent_step < num_descent_steps; ++descent_step)
        {
            dpl::copy(q_policy, dpl::begin(descent_data.gradient),
                      dpl::end(descent_data.gradient), dpl::begin(descent_data.gradient_prev));

            compute_descent_gradient(q, Psi, dt, gravity, node_data.predicted_positions, descent_data.gradient);

            dpl::copy(q_policy, dpl::begin(descent_data.descent_direction),
                      dpl::end(descent_data.descent_direction), dpl::begin(descent_data.descent_direction_prev));
            set_descent_direction(q);

            compute_directional_hessian(q, Psi, dt, gravity);

            initial_step(q, dt);

            back_trace_line_search(q, Psi, dt, gravity);

            update_particle_deformation_gradients_implicit(q, dt);
        }

        compute_node_velocities_implicit(q, dt);
        transfer_velocity_nodes_to_particles_APIC(q);

        apply_particle_velocities_with_aether_damping(q, dt, mu_step);


    }
    q.wait();

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
         sycl::accessor descent_gradient_acc(gradient_destination, h);
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
template<typename ConstitutiveModel>
void Engine<TInterpolationScheme>::compute_descent_value(
        sycl::queue &q,
        const ConstitutiveModel Psi,
        scalar_t dt,
        const double gravity,
        sycl::buffer<Coordinate_t> &node_positions,
        sycl::buffer<scalar_t> &value_destination,
        sycl::buffer<bool>& continue_line_search_flag
)
{
    auto interaction_access = pgi_manager.kernel_accessor;
    auto n = interpolator;
    //MPM forces
    q.submit([&](sycl::handler &h)
     {
         sycl::accessor deformation_gradient_acc(particle_data.deformation_gradients, h);
         sycl::accessor rest_volume_acc(particle_data.rest_volumes, h);
         sycl::accessor continue_line_search_flag_acc(descent_data.continue_line_search_flag, h);

         interaction_access.give_kernel_access(h);

         h.parallel_for(
             particle_data.particle_count,
             sycl::reduction(
                     value_destination,
                     h,
                     std::plus<scalar_t>(),
                     {sycl::property_list {sycl::property::reduction::initialize_to_identity()}}
                     ),
             [=](sycl::id<1> idx, auto& sum)
             {
                 if(!continue_line_search_flag_acc[0])
                     return;
                 size_t pid = idx[0];
                 scalar_t V_p = rest_volume_acc[pid];

                 CoordinateMatrix_t F = deformation_gradient_acc[pid];
                 scalar_t psi_value = Psi.value(F);

                 sum += V_p * psi_value;
             });
     });


    //inertia + walls + gravity
    q.submit([&](sycl::handler &h)
     {
         sycl::accessor node_mass_acc(node_data.masses, h);
         sycl::accessor node_predicted_positions_acc(node_positions, h);
         sycl::accessor node_inertial_positions_acc(node_data.inertial_positions, h);
         sycl::accessor walls_acc(collision_walls, h);
         sycl::accessor continue_line_search_flag_acc(descent_data.continue_line_search_flag, h);

         interaction_access.give_kernel_access(h);

         h.parallel_for(
                 node_data.max_node_count,
                 sycl::reduction(
                         value_destination,
                         h,
                         std::plus<scalar_t>(),
                         {}
                 ),
                 [=](sycl::id<1> idx, auto& sum)
                 {
                     if(!continue_line_search_flag_acc[0])
                         return;
                     const size_t node_count = interaction_access.node_count();
                     const size_t node_id = idx[0];
                     if (node_id >= node_count)
                         return;

                     Coordinate_t gravity_vec = Coordinate_t(0.0, gravity);
                     NodeIndex_t node_index = interaction_access.get_node_index(node_id);
                     scalar_t mass_i = node_mass_acc[node_id];
                     Coordinate_t inertial_position = node_inertial_positions_acc[node_id];
                     Coordinate_t node_predicted_position = node_predicted_positions_acc[node_id];

                     sum += mass_i * gravity_vec.dot(node_predicted_position);
                     sum += mass_i / (2.0 * dt * dt) * (node_predicted_position - inertial_position).dot(node_predicted_position - inertial_position);

                     for (ElasticCollisionWall<CoordinateConfiguration> &wall: walls_acc)
                     {
                         sum += wall.value(node_predicted_position);
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

    initial_vec_dot(q, node_data.max_node_count, pgi_manager.node_count,
                    descent_data.gradient, descent_data.gradient, descent_data.gradient_dot2);
    initial_vec_dot(q, node_data.max_node_count, pgi_manager.node_count,
                    descent_data.gradient, descent_data.gradient_prev, descent_data.gradient_dot_gradient_prev);
    initial_vec_dot(q, node_data.max_node_count, pgi_manager.node_count,
                    descent_data.gradient_prev, descent_data.gradient_prev, descent_data.gradient_prev_dot2);



    q.submit([&](sycl::handler &h)
     {
         sycl::accessor gradient_dot2_acc(descent_data.gradient_dot2, h);
         sycl::accessor gradient_dot_gradient_prev_acc(descent_data.gradient_dot_gradient_prev, h);
         sycl::accessor gradient_prev_dot2_acc(descent_data.gradient_prev_dot2, h);
         sycl::accessor beta_acc(descent_data.beta_fletcher_reeves, h);
         sycl::accessor descent_step_reset_counter_acc(descent_data.descent_step_reset_counter, h);

         interaction_access.give_kernel_access(h);

         h.single_task([=](){
            scalar_t beta = 0.0;
            if(descent_step_reset_counter_acc[0] == 0)
            {
                beta = 0.0;
            }
            else
            {
                beta = (gradient_dot2_acc[0] - gradient_dot_gradient_prev_acc[0]) / gradient_prev_dot2_acc[0];
                beta = std::max(beta, 0.0);
                if (std::isnan(beta))
                    beta = 0.0;
            }
             descent_step_reset_counter_acc[0]++;
            int max_reset_steps = 10;
            if(descent_step_reset_counter_acc[0] >= max_reset_steps)
                descent_step_reset_counter_acc[0] = 0;
             beta_acc[0] = beta;
         });
     });

    q.submit([&](sycl::handler &h)
     {
         sycl::accessor descent_gradient_acc(descent_data.gradient, h);
         sycl::accessor descent_direction_acc(descent_data.descent_direction, h);
         sycl::accessor descent_direction_prev_acc(descent_data.descent_direction_prev, h);
         sycl::accessor beta_acc(descent_data.beta_fletcher_reeves, h);


         interaction_access.give_kernel_access(h);

         h.parallel_for(node_data.max_node_count, [=](sycl::id<1> idx)
         {
             const size_t node_count = interaction_access.node_count();
             const size_t node_id = idx[0];
             if (node_id >= node_count)
                 return;
             descent_direction_acc[node_id] = -descent_gradient_acc[node_id] + beta_acc[0] * descent_direction_prev_acc[node_id];
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
    initial_vec_dot(q, node_data.max_node_count, pgi_manager.node_count, descent_data.descent_direction, descent_data.descent_direction, descent_data.descent_direction_dot2);

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
        sycl::queue &q, const scalar_t dt)
{

    auto interaction_access = pgi_manager.kernel_accessor;

    q.submit([&](sycl::handler& h)
     {
         sycl::accessor directional_hessian_acc(descent_data.directional_hessian, h);
         sycl::accessor descent_direction_dot_grad_acc(descent_data.descent_direction_dot_grad, h);
         sycl::accessor descent_direction_dot2_acc(descent_data.descent_direction_dot2, h);
         sycl::accessor alpha_step_acc(descent_data.alpha_step, h);

         interaction_access.give_kernel_access(h);

         h.single_task([=]()
           {
               scalar_t delta_x_norm = std::sqrt(descent_direction_dot2_acc[0]);
               scalar_t directional_hessian = directional_hessian_acc[0];
               scalar_t directional_gradient = descent_direction_dot_grad_acc[0];

               if(delta_x_norm < 1.0E-3 || std::abs(directional_gradient) < 1.0E-3)
               {
                   alpha_step_acc[0] = 0.0;
                   return;
               }


               scalar_t max_delta_v = 500.0;
               //the following formula limits the root mean square of the delta_node_vector / dt,
               //i.e. the RMS of the velocity change of the nodes
               scalar_t max_alpha = max_delta_v * dt * std::sqrt(interaction_access.node_count()) / delta_x_norm;
               scalar_t alpha = - directional_gradient / directional_hessian;

               if(std::isnan(alpha) || alpha > max_alpha)
                   alpha = max_alpha;

               alpha_step_acc[0] =  alpha;
           });
     });

    q.submit([&](sycl::handler &h)
     {
         sycl::accessor node_positions_acc(node_data.predicted_positions, h);
         sycl::accessor node_line_search_positions_acc(descent_data.node_line_search_positions, h);
         sycl::accessor alpha_step_acc(descent_data.alpha_step, h);
         sycl::accessor descent_direction_acc(descent_data.descent_direction, h);

         interaction_access.give_kernel_access(h);

         h.parallel_for(node_data.max_node_count, [=](sycl::id<1> idx)
         {
             const size_t node_count = interaction_access.node_count();
             const size_t node_id = idx[0];
             if (node_id >= node_count)
                 return;
             node_line_search_positions_acc[node_id] =
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


template<class TInterpolationScheme>
template<typename ConstitutiveModel>
void Engine<TInterpolationScheme>::back_trace_line_search(
        sycl::queue &q,
        const ConstitutiveModel Psi,
        scalar_t dt,
        const double gravity
)
{
    auto q_policy = dpl::execution::make_device_policy(q);

    auto interaction_access = pgi_manager.kernel_accessor;
    scalar_t rho = 0.5;
    q.submit([&](sycl::handler& h){
        sycl::accessor continue_line_search_flag_acc(descent_data.continue_line_search_flag, h);
        sycl::accessor multiplier_acc(descent_data.line_search_multiplier, h);
        h.single_task([=](){
            continue_line_search_flag_acc[0] = true;

            multiplier_acc[0] = 1.0;});
    });

    compute_descent_value(q, Psi, dt, gravity, node_data.predicted_positions, descent_data.descent_value_0, descent_data.continue_line_search_flag);
    int max_backtrace_steps = 20;

    for(int i = 0; i < max_backtrace_steps; ++i)
    {
        compute_descent_value(q, Psi, dt, gravity, descent_data.node_line_search_positions, descent_data.descent_value,
                              descent_data.continue_line_search_flag);

        q.submit([&](sycl::handler& h){
            sycl::accessor continue_line_search_flag_acc(descent_data.continue_line_search_flag, h);
            sycl::accessor descent_value_0_acc(descent_data.descent_value_0, h);
            sycl::accessor descent_value_acc(descent_data.descent_value, h);
            sycl::accessor multiplier_acc(descent_data.line_search_multiplier, h);

            h.single_task([=](){
                if(!continue_line_search_flag_acc[0] )
                    return;
                if(descent_value_acc[0] < descent_value_0_acc[0])
                {
                    continue_line_search_flag_acc[0] = false;
                    return;
                }
                multiplier_acc[0] = rho * multiplier_acc[0];

            });
        });

        q.submit([&](sycl::handler& h){
            sycl::accessor descent_value_0_acc(descent_data.descent_value_0, h);
            sycl::accessor descent_value_acc(descent_data.descent_value, h);
            sycl::accessor alpha_step_acc(descent_data.alpha_step, h);
            sycl::accessor node_positions_acc(node_data.predicted_positions, h);
            sycl::accessor node_line_search_positions_acc(descent_data.node_line_search_positions, h);
            sycl::accessor descent_direction_acc(descent_data.descent_direction, h);
            sycl::accessor multiplier_acc(descent_data.line_search_multiplier, h);
            sycl::accessor continue_line_search_flag_acc(descent_data.continue_line_search_flag, h);

            interaction_access.give_kernel_access(h);

            h.parallel_for(node_data.max_node_count,[=](sycl::id<1> idx)
            {
                if(!continue_line_search_flag_acc[0] )
                    return;
                const size_t node_count = interaction_access.node_count();
                const size_t node_id = idx[0];
                if(node_id >= node_count)
                    return;

                node_line_search_positions_acc[node_id] =
                        node_positions_acc[node_id] + multiplier_acc[0] * alpha_step_acc[0] * descent_direction_acc[node_id];

            });
        });



    }
    dpl::copy(q_policy, dpl::begin(descent_data.node_line_search_positions), dpl::end(descent_data.node_line_search_positions),
              dpl::begin(node_data.predicted_positions));

}

}

#endif //DESCENT_IMPL_HPP