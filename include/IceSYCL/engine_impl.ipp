//
// Created by robert-denomme on 8/27/24.
//
#ifndef ENGINE_IMPL_HPP
#define ENGINE_IMPL_HPP


namespace iceSYCL
{

template<class TInterpolationScheme>
template<typename ConstitutiveModel>
void Engine<TInterpolationScheme>::step_frame(const ConstitutiveModel Psi)
{
    sycl::queue q{};
    auto q_policy = dpl::execution::make_device_policy(q);
    const size_t num_steps_per_frame = 50;
    const scalar_t dt = 1.0 / (50 * num_steps_per_frame);

    for(size_t step = 0; step < num_steps_per_frame; ++step)
    {
        dpl::copy(q_policy, dpl::begin(particle_data.positions), dpl::end(particle_data.positions), dpl::begin(particle_data.positions_prev));
        dpl::copy(q_policy, dpl::begin(particle_data.velocities), dpl::end(particle_data.velocities), dpl::begin(particle_data.velocities_prev));

        pgi_manager.update_particle_locations(q, particle_data.positions, interpolator);

        transer_mass_particles_to_nodes(q);
        transfer_momentum_particles_to_nodes_APIC(q);
        apply_particle_forces_to_grid(q, collision_walls, dt);
        apply_mpm_hyperelastic_forces_to_grid(q, Psi, dt);
        compute_node_velocities(q);
        transfer_velocity_nodes_to_particles_APIC(q);

        dpl::copy(q_policy, dpl::begin(particle_data.deformation_gradients), dpl::end(particle_data.deformation_gradients), dpl::begin(particle_data.deformation_gradients_prev));

        update_particle_deformation_gradients(q, dt);

        q.submit([&](sycl::handler& h)
        {
            sycl::accessor positions_acc(particle_data.positions, h);
            sycl::accessor velocities_acc(particle_data.velocities, h);

            h.parallel_for(particle_count, [=](sycl::id<1> idx)
            {
                size_t pid = idx[0];
                positions_acc[pid] = positions_acc[pid] + dt * velocities_acc[pid];
            });
        });
    }
    q.wait();
}

template<class TInterpolationScheme>
void Engine<TInterpolationScheme>::apply_particle_forces_to_grid(sycl::queue& q, sycl::buffer<ElasticCollisionWall<CoordinateConfiguration>>& walls, scalar_t dt)
{
    auto interaction_access = pgi_manager.kernel_accessor;
    auto n = interpolator;
    //nodes
    q.submit([&](sycl::handler& h)
    {
        sycl::accessor particle_mass_acc(particle_data.masses, h);
        sycl::accessor particle_positions_acc(particle_data.positions, h);
        sycl::accessor node_momenta_acc(node_data.momenta, h);
        sycl::accessor walls_acc(walls, h);

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
                scalar_t mass_p = particle_mass_acc[pid];
                Coordinate_t x_p = particle_positions_acc[pid];

                NodeIndex_t node_index = interaction.node_index;

                for(ElasticCollisionWall<CoordinateConfiguration>& wall : walls_acc)
                {
                    force_i -= n.value(node_index, x_p) * wall.gradient(x_p);
                }

                Coordinate_t gravity = Coordinate_t(0.0, -981.0);
                force_i += n.value(node_index, x_p) * mass_p * gravity;
            }

            node_momenta_acc[node_id] += dt * force_i;
        });
    });
}

template<class TInterpolationScheme>
void Engine<TInterpolationScheme>::transer_mass_particles_to_nodes(sycl::queue& q)
{
    particle_grid_operations::transfer_data_particles_to_grid(q,
        pgi_manager,
        interpolator,
        particle_data.masses,
        node_data.masses,
        particle_data.positions,
        0.0);
}

template<class TInterpolationScheme>
void Engine<TInterpolationScheme>::transfer_momentum_particles_to_nodes_APIC(sycl::queue& q)
{

    auto interaction_access = pgi_manager.kernel_accessor;

    auto n = interpolator;
    //particles
    q.submit([&](sycl::handler& h)
    {
        sycl::accessor particle_mass_acc(particle_data.masses, h);
        sycl::accessor particle_positions_acc(particle_data.positions, h);
        sycl::accessor particle_D_matrices_acc(particle_data.D_matrices, h);
        interaction_access.give_kernel_access(h);

        h.parallel_for(particle_count,[=](sycl::id<1> idx)
        {
            size_t pid = idx[0];

            Coordinate_t x_p = particle_positions_acc[pid];

            CoordinateMatrix_t D_p = CoordinateMatrix_t::Zero();
            for(auto interaction_it = interaction_access.particle_interactions_begin(pid);
                interaction_it != interaction_access.particle_interactions_end(pid);
                ++interaction_it)
            {
                ParticleNodeInteraction<typename TInterpolationScheme::CoordinateConfiguration> interaction = *interaction_it;


                NodeIndex_t node_index = interaction.node_index;
                scalar_t n_value = n.value(node_index, x_p);

                D_p += n_value * (n.position(node_index) - x_p) * (n.position(node_index) - x_p).transpose();
            }

            particle_D_matrices_acc[pid] = D_p;
        });
    });

    //nodes
    q.submit([&](sycl::handler& h)
    {
        sycl::accessor particle_mass_acc(particle_data.masses, h);
        sycl::accessor particle_positions_acc(particle_data.positions, h);
        sycl::accessor particle_velocity_acc(particle_data.velocities, h);
        sycl::accessor particle_D_matrices_acc(particle_data.D_matrices, h);
        sycl::accessor particle_B_matrices_acc(particle_data.B_matrices, h);
        sycl::accessor node_momenta_acc(node_data.momenta, h);

        interaction_access.give_kernel_access(h);

        h.parallel_for(node_data.max_node_count,[=](sycl::id<1> idx)
        {
            const size_t node_count = interaction_access.node_count();
            const size_t node_id = idx[0];
            if(node_id >= node_count)
                return;

            Coordinate_t momentum_i = Coordinate_t::Zero();
            for(auto interaction_it = interaction_access.node_interactions_begin(node_id);
                interaction_it != interaction_access.node_interactions_end(node_id);
                ++interaction_it)
            {
                ParticleNodeInteraction<typename TInterpolationScheme::CoordinateConfiguration> interaction = *interaction_it;
                size_t pid = interaction.particle_id;
                scalar_t mass_p = particle_mass_acc[pid];
                Coordinate_t x_p = particle_positions_acc[pid];
                Coordinate_t v_p = particle_velocity_acc[pid];
                CoordinateMatrix_t B_p = particle_B_matrices_acc[pid];
                CoordinateMatrix_t D_p = particle_D_matrices_acc[pid];

                NodeIndex_t node_index = interaction.node_index;

                momentum_i += n.value(node_index, x_p) * mass_p * (v_p + B_p * small_la::inverse(D_p) * (n.position(node_index) - x_p));
            }

            node_momenta_acc[node_id] = momentum_i;
        });
    });
}


template<class TInterpolationScheme>
void Engine<TInterpolationScheme>::compute_node_velocities(sycl::queue& q)
{

    //nodes
    q.submit([&](sycl::handler& h)
    {
        sycl::accessor node_mass_acc(node_data.masses, h);
        sycl::accessor node_velocity_acc(node_data.velocities, h);
        sycl::accessor node_momenta_acc(node_data.momenta, h);
        sycl::accessor node_count_acc(pgi_manager.node_count, h);

        h.parallel_for(node_data.max_node_count,[=](sycl::id<1> idx)
        {
            const size_t node_count = node_count_acc[0];
            const size_t node_id = idx[0];
            if(node_id >= node_count)
                return;


            Coordinate_t momentum_i = node_momenta_acc[node_id];
            scalar_t mass_i = node_mass_acc[node_id];

            Coordinate_t velocity_i = Coordinate_t::Zero();
            if(mass_i > 0.0)
                velocity_i = 1.0 / (mass_i) * momentum_i;

            node_velocity_acc[node_id] = velocity_i;
        });
    });
}

template<class TInterpolationScheme>
void Engine<TInterpolationScheme>::transfer_velocity_nodes_to_particles(sycl::queue& q)
{
    auto n = interpolator;
    auto interaction_access = pgi_manager.kernel_accessor;
    q.submit([&](sycl::handler& h)
    {
        sycl::accessor node_velocity_acc(node_data.velocities, h);
        sycl::accessor particle_velocity_acc(particle_data.velocities, h);
        sycl::accessor particle_positions_acc(particle_data.positions, h);

        interaction_access.give_kernel_access(h);

        h.parallel_for(particle_count,[=](sycl::id<1> idx)
        {
            size_t pid = idx[0];
            Coordinate_t x_p = particle_positions_acc[pid];

            Coordinate_t v_p = Coordinate_t::Zero();
            for(auto interaction_it = interaction_access.particle_interactions_begin(pid);
                interaction_it != interaction_access.particle_interactions_end(pid);
                ++interaction_it)
            {
                ParticleNodeInteraction<typename TInterpolationScheme::CoordinateConfiguration> interaction = *interaction_it;
                size_t node_id = interaction.node_id;
                Coordinate_t v_i = node_velocity_acc[node_id];
                NodeIndex_t node_index = interaction.node_index;

                v_p += n.value(node_index, x_p) * v_i;
            }

            particle_velocity_acc[pid] = v_p;
        });
    });

}

template<class TInterpolationScheme>
void Engine<TInterpolationScheme>::transfer_velocity_nodes_to_particles_APIC(sycl::queue& q)
{
    transfer_velocity_nodes_to_particles(q);

    auto n = interpolator;
    auto interaction_access = pgi_manager.kernel_accessor;
    q.submit([&](sycl::handler& h)
    {
        sycl::accessor node_velocity_acc(node_data.velocities, h);
        sycl::accessor particle_positions_acc(particle_data.positions, h);
        sycl::accessor particle_B_matrices_acc(particle_data.B_matrices, h);

        interaction_access.give_kernel_access(h);

        h.parallel_for(particle_count,[=](sycl::id<1> idx)
        {
            size_t pid = idx[0];
            Coordinate_t x_p = particle_positions_acc[pid];

            CoordinateMatrix_t B_p = CoordinateMatrix_t::Zero();
            for(auto interaction_it = interaction_access.particle_interactions_begin(pid);
                interaction_it != interaction_access.particle_interactions_end(pid);
                ++interaction_it)
            {
                ParticleNodeInteraction<typename TInterpolationScheme::CoordinateConfiguration> interaction = *interaction_it;
                size_t node_id = interaction.node_id;
                Coordinate_t v_i = node_velocity_acc[node_id];
                NodeIndex_t node_index = interaction.node_index;
                Coordinate_t x_i = n.position(node_index);

                B_p += n.value(node_index, x_p) * v_i * (x_i - x_p).transpose();
            }

            particle_B_matrices_acc[pid] = B_p;
        });
    });

}

template<class TInterpolationScheme>
void Engine<TInterpolationScheme>::update_particle_deformation_gradients(sycl::queue& q, Engine<TInterpolationScheme>::scalar_t dt)
{
    auto n = interpolator;
    auto interaction_access = pgi_manager.kernel_accessor;
    q.submit([&](sycl::handler& h)
    {
        sycl::accessor del_v_del_x_acc(particle_data.del_v_del_x, h);
        sycl::accessor particle_positions_acc(particle_data.positions, h);
        sycl::accessor node_velocity_acc(node_data.velocities, h);

        interaction_access.give_kernel_access(h);

        h.parallel_for(particle_count,[=](sycl::id<1> idx)
        {
            size_t pid = idx[0];
            Coordinate_t x_p = particle_positions_acc[pid];

            CoordinateMatrix_t del_v_del_x_p = CoordinateMatrix_t::Identity();
            for(auto interaction_it = interaction_access.particle_interactions_begin(pid);
                interaction_it != interaction_access.particle_interactions_end(pid);
                ++interaction_it)
            {
                ParticleNodeInteraction<typename TInterpolationScheme::CoordinateConfiguration> interaction = *interaction_it;
                size_t node_id = interaction.node_id;
                Coordinate_t v_i = node_velocity_acc[node_id];
                NodeIndex_t node_index = interaction.node_index;
                Coordinate_t x_i = n.position(node_index);

                Coordinate_t grad_n_i = n.gradient(node_index, x_p);

                del_v_del_x_p += dt * v_i * grad_n_i.transpose();
            }
            del_v_del_x_acc[pid] = del_v_del_x_p;
        });
    });
}


template<typename TInterpolationScheme>
template<typename ConstitutiveModel>
void Engine<TInterpolationScheme>::apply_mpm_hyperelastic_forces_to_grid(sycl::queue& q, const ConstitutiveModel Psi, const scalar_t dt)
{
    auto interaction_access = pgi_manager.kernel_accessor;
    auto n = interpolator;
    //nodes
    q.submit([&](sycl::handler& h)
    {
         sycl::accessor particle_mass_acc(particle_data.masses, h);
         sycl::accessor particle_positions_acc(particle_data.positions, h);
         sycl::accessor node_momenta_acc(node_data.momenta, h);
         sycl::accessor deformation_gradient_acc(particle_data.deformation_gradients, h);
         sycl::accessor rest_volume_acc(particle_data.rest_volumes, h);

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
                 scalar_t mass_p = particle_mass_acc[pid];
                 Coordinate_t x_p = particle_positions_acc[pid];
                 scalar_t V_p = rest_volume_acc[pid];
                 CoordinateMatrix_t F = deformation_gradient_acc[pid];
                 CoordinateMatrix_t PK = Psi.PK(F);

                 NodeIndex_t node_index = interaction.node_index;

                 force_i -= dt * V_p * PK * F.transpose() * n.gradient(node_index, x_p);

             }

             node_momenta_acc[node_id] += dt * force_i;
         });
    });
}
}

#endif //ENGINE_IMPL_HPP