//
// Created by robert-denomme on 8/27/24.
//
#ifndef LCG_ENGINE_IMPL_HPP
#define LCG_ENGINE_IMPL_HPP


namespace iceSYCL
{


template<class TInterpolationScheme>
template<typename ConstitutiveModel>
void Engine<TInterpolationScheme>::step_frame_lcg_implicit(const ConstitutiveModel Psi,
                                                       const size_t num_steps_per_frame,
                                                       const size_t num_descent_steps,
                                                       const size_t num_linear_solve_steps,
                                                       const size_t max_num_backsteps,
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

    HessianApplier<ConstitutiveModel> hessian_applier(this, Psi, dt, gravity);

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

        update_particle_deformation_gradients_implicit(q, dt, node_data.predicted_positions);

        for(size_t descent_step = 0; descent_step < num_descent_steps; ++descent_step)
        {

            compute_descent_gradient(q, Psi, dt, gravity, node_data.predicted_positions, descent_data.gradient);

            q.submit([&](sycl::handler& h){
                sycl::accessor gradient_acc(descent_data.gradient, h);
                sycl::accessor descent_direction_acc(descent_data.descent_direction, h);
                sycl::accessor actual_vec_size_acc(pgi_manager.node_count, h);

                h.parallel_for(node_data.max_node_count, [=](sycl::id<1> idx)
                {
                    const size_t actual_vec_size = actual_vec_size_acc[0];
                    const size_t i = idx[0];
                    if (i >= actual_vec_size)
                        return;
                    descent_direction_acc[i] = -gradient_acc[i];
                });
            });

            if(num_linear_solve_steps > 0)
            {
                lcg_solver.Solve(q, pgi_manager.node_count, hessian_applier, descent_data.descent_direction,
                                 descent_data.descent_direction, num_linear_solve_steps);
            }
            compute_directional_hessian(q, Psi, dt, gravity);

            initial_step(q, dt);
            back_trace_line_search(q, Psi, max_num_backsteps, dt, gravity);

            update_particle_deformation_gradients_implicit(q, dt, node_data.predicted_positions);
        }

        compute_node_velocities_implicit(q, dt);
        transfer_velocity_nodes_to_particles_APIC(q);

        apply_particle_velocities_with_aether_damping(q, dt, mu_step);


    }
    q.wait();

}


}

#endif //LCG_ENGINE_IMPL_HPP