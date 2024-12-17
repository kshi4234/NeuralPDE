finite_diff_GP.py fits only to the finite difference points, without using PINN as the mean function.

mean_PINN.py fits to finite difference points while using the PINN as the prior mean function (I suspect this is not super interesting)

mean_PINN_test.py fits to finite difference points, then uses the PINN as the predictive mean.
