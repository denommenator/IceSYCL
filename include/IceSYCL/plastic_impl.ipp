//
// Created by robert-denomme on 8/27/24.
//
#ifndef PLASTIC_IMPL_HPP
#define PLASTIC_IMPL_HPP


namespace iceSYCL
{

template<class TInterpolationScheme>
template<typename PlasticConstitutiveModel>
void Engine<TInterpolationScheme>::step_frame_plastic_explicit(sycl::buffer<PlasticConstitutiveModel> Psis, const size_t num_steps_per_frame, const double mu_velocity_damping, const double gravity)
{
    sycl::queue q{};
    auto q_policy = dpl::execution::make_device_policy(q);
    const scalar_t dt = 1.0 / (50 * num_steps_per_frame);
    //dt = 1/N => mu_step^N = mu_step^(1/dt) = mu
    //mu_step = mu^(dt)
    const scalar_t mu_step = std::pow(mu_velocity_damping, dt);

    for(size_t step = 0; step < num_steps_per_frame; ++step)
    {
        dpl::copy(q_policy, dpl::begin(particle_data.positions), dpl::end(particle_data.positions), dpl::begin(particle_data.positions_prev));
        dpl::copy(q_policy, dpl::begin(particle_data.velocities), dpl::end(particle_data.velocities), dpl::begin(particle_data.velocities_prev));

        pgi_manager.update_particle_locations(q, particle_data.positions, interpolator);

        transer_mass_particles_to_nodes(q);
        transfer_momentum_particles_to_nodes_APIC(q);
        apply_particle_forces_to_grid(q, collision_walls, dt, gravity);
        apply_mpm_elastoplastic_forces_to_grid(q, Psis, dt);
        compute_node_velocities(q);
        transfer_velocity_nodes_to_particles_APIC(q);

        dpl::copy(q_policy, dpl::begin(particle_data.deformation_gradients), dpl::end(particle_data.deformation_gradients), dpl::begin(particle_data.deformation_gradients_prev));

        update_particle_deformation_gradients(q, dt);

        apply_particle_velocities_with_aether_damping(q, dt, mu_step);
    }
    q.wait();
}

template<typename TInterpolationScheme>
template<typename PlasticConstitutiveModel>
void Engine<TInterpolationScheme>::apply_mpm_elastoplastic_forces_to_grid(sycl::queue& q, sycl::buffer<PlasticConstitutiveModel> Psis, const scalar_t dt)
{
    q.submit([&](sycl::handler &h)
     {
         sycl::accessor deformation_gradients_acc(particle_data.deformation_gradients, h);
         sycl::accessor Psis_acc(Psis, h);

         h.parallel_for(particle_count, [=](sycl::id<1> idx)
         {
             size_t pid = idx[0];
             CoordinateMatrix_t F = deformation_gradients_acc[pid];
             Psis_acc[pid].update(F);
         });
     });
    q.wait();

    auto interaction_access = pgi_manager.kernel_accessor;
    auto n = interpolator;
    //nodes
    q.submit([&](sycl::handler& h)
    {
         sycl::accessor particle_positions_acc(particle_data.positions, h);
         sycl::accessor node_momenta_acc(node_data.momenta, h);
         sycl::accessor rest_volume_acc(particle_data.rest_volumes, h);
        sycl::accessor deformation_gradient_acc(particle_data.deformation_gradients, h);
         sycl::accessor Psis_acc(Psis, h);

         interaction_access.give_kernel_access(h);

         h.parallel_for(node_data.max_node_count,[=](sycl::id<1> idx)
         {
             const size_t node_count = interaction_access.node_count();
             const size_t node_id = idx[0];
             if(node_id >= node_count)
                 return;

             Coordinate_t force_i = Coordinate_t::Zero();
             for(auto interaction_it = interaction_access.node_interactions_begin(node_id);
                 interaction_it != interaction_access.node_interactions_end(node_id);
                 ++interaction_it)
             {
                 ParticleNodeInteraction<typename TInterpolationScheme::CoordinateConfiguration> interaction = *interaction_it;
                 size_t pid = interaction.particle_id;
                 Coordinate_t x_p = particle_positions_acc[pid];
                 scalar_t V_p = rest_volume_acc[pid];
                 CoordinateMatrix_t F = deformation_gradient_acc[pid];
                 const PlasticConstitutiveModel& Psi_p = Psis_acc[pid];
                 CoordinateMatrix_t PK = Psi_p.PK();

                 NodeIndex_t node_index = interaction.node_index;

                 force_i -= V_p * PK * Psi_p.F_E.transpose() * n.gradient(node_index, x_p);

             }

             node_momenta_acc[node_id] += dt * force_i;
         });
    });
}
}

#endif //PLASTIC_IMPL_HPP