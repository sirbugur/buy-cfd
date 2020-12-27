
static char help[] = "Solves Stokes / linearized Navier-Stokes system with KSP or MUMPS.\n\n";

#include <petscksp.h>
#include <petscvec.h>
#include <iostream>
#include <string>
#include <matplot/matplot.h>

void plot(matplot::vector_2d &, int, int);

int main(int argc, char **args)
{
    PetscMPIInt size;
    PetscErrorCode ierr;
    PetscInt i, j, Imax = 5, Jmax = 5, m, n;
    PetscInt ind_u, ind_v; // local matrix indices
    PetscInt rstart, rend, nlocal;
    PetscInt nu, nv, np, nt;
    PetscReal dx, dy;
    PetscReal y, ic;
    PetscViewer viewer;
    PetscInt deltaRe = 50, latestRe = 10;
    int Re = 0;

    KSP ksp; // linear solver context
    PC pc;
    Mat A;
    Vec RHS, x;
    PetscInt columns[7];      // column indexes
    PetscScalar values[7];    // corresponding values for col
    PetscInt barInd[4];       // Ubar and Vbar indices in x vector
    PetscScalar barValues[4]; // Ubar and Vbar values gotten from x vector
    PetscScalar x_getm;       // Get value of mth row from x vector
    PetscScalar ZERO = 0;

    PetscReal norm;

    PetscScalar *x_arr;
    PetscBool verbose = PETSC_FALSE;

    ierr = PetscInitialize(&argc, &args, (char *)0, help);
    if (ierr)
        return ierr;
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);
    CHKERRQ(ierr);

    // Get Imax and Jmax from command line:
    PetscOptionsGetInt(NULL, NULL, "-Imax", &Imax, NULL);
    PetscOptionsGetInt(NULL, NULL, "-Jmax", &Jmax, NULL);

    // Get latest Reynolds number computed and delta Re
    PetscOptionsGetInt(NULL, NULL, "-deltaRe", &deltaRe, NULL);
    PetscOptionsGetInt(NULL, NULL, "-latestRe", &latestRe, NULL);

    nu = Imax * (Jmax - 1);       // number of u
    nv = Jmax * (Imax - 1);       // number of v
    np = (Jmax - 1) * (Imax - 1); // number of P
    nt = nu + nv + np;            // total number of matrix
    dx = 5.0 / (Imax - 1);        // here 5.0 means 5 unit in x-length
    dy = 1.0 / (Jmax - 1);        // here 1.0 means 1 unit in y-length

    std::cout << "nu, nv, np, dx, dy:\t" << nu << ", " << nv << ", " << np << ", " << dx << ", " << dy << std::endl;

    /*
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed. For this simple case let PETSc decide how
     many elements of the vector are stored on each processor. The second
     argument to VecSetSizes() below causes PETSc to decide.
    */
    ierr = VecCreate(PETSC_COMM_WORLD, &x);
    CHKERRQ(ierr);
    ierr = VecSetSizes(x, PETSC_DECIDE, nt);
    CHKERRQ(ierr);
    ierr = VecSetFromOptions(x);
    CHKERRQ(ierr);
    ierr = VecDuplicate(x, &RHS);
    CHKERRQ(ierr);

    /* Identify the starting and ending mesh points on each
     processor for the interior part of the mesh. We let PETSc decide
     above. */

    ierr = VecGetOwnershipRange(x, &rstart, &rend);
    CHKERRQ(ierr);
    ierr = VecGetLocalSize(x, &nlocal);
    CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_WORLD, &A);
    CHKERRQ(ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, nt, nt);
    CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);
    CHKERRQ(ierr);
    ierr = MatSetUp(A);
    CHKERRQ(ierr);

    while (Re <= latestRe)
    {
        // x-momentum equation
        for (i = 0; i < Imax; i++)
        {
            for (j = 0; j < Jmax - 1; j++)
            {
                // Row index of current u
                m = j * Imax + i;
                //       index of P      + number of total u & v
                n = (j * (Imax - 1) + i) + nu + nv;

                if (i == 0) // Left Edge of the Domain
                {
                    MatSetValue(A, m, m, 1, INSERT_VALUES);
                    if (dy * j + dy / 2 > 0.5) // velocity inlet of the left edge
                    {
                        y = j * dy + dy / 2;
                        ic = -24 * (1 - y) * (0.5 - y);
                        VecSetValue(RHS, m, ic, INSERT_VALUES);
                    }
                }

                else if (i == Imax - 1) // pressure outlet of the domain
                {
                    columns[0] = m;     // actual index
                    columns[1] = m - 1; // left index
                    columns[2] = n - 1; // Pressure index

                    values[0] = 1.0 / dx;
                    values[1] = -1.0 / dx;
                    values[2] = -1;

                    ierr = MatSetValues(A, 1, &m, 3, columns, values, INSERT_VALUES);
                }
                else // Bottom Edge, Top Edge and Inner Domain
                {
                    // Handling non-linear term
                    PetscInt m_Vbar = (j) * (Imax - 1) + i + nu; // row index of v_{i,j}^n
                    // Finding Indices for \Bar{V}
                    barInd[0] = m_Vbar;
                    barInd[1] = m_Vbar - 1;
                    barInd[2] = m_Vbar + Imax - 1;
                    barInd[3] = m_Vbar + Imax - 2;
                    // Getting Values from x vector
                    VecGetValues(x, 4, barInd, barValues);
                    PetscScalar Vbar; // total value
                    for (int i = 0; i < 4; i++)
                    {
                        Vbar += barValues[i];
                    }
                    Vbar /= 4.0;
                    VecGetValues(x, 1, &m, &x_getm);

                    if (j == 0) // Bottom Edge of Domain
                    {
                        columns[0] = m;        // actual index
                        columns[1] = m - 1;    // left index
                        columns[2] = m + 1;    // right index
                        columns[3] = m + Imax; // north index
                        columns[4] = n - 1;    // P_{i-1,j}
                        columns[5] = n;        // P_{i,j}

                        values[0] = 2.0 / pow(dx, 2) + 3.0 / pow(dy, 2) - Re * Vbar / (2 * dy);
                        values[1] = -1.0 / pow(dx, 2) - Re * x_getm / (2 * dx);
                        values[2] = -1.0 / pow(dx, 2) + Re * x_getm / (2 * dx);
                        values[3] = -1.0 / pow(dy, 2) + Re * Vbar / (2 * dy);
                        values[4] = -1.0 / dx;
                        values[5] = 1.0 / dx;

                        ierr = MatSetValues(A, 1, &m, 6, columns, values, INSERT_VALUES);
                    }
                    else if (j == Jmax - 2) // Top Edge of Domain
                    {
                        columns[0] = m;        // actual index
                        columns[1] = m - 1;    // left index
                        columns[2] = m + 1;    // right index
                        columns[3] = m - Imax; // south index
                        columns[4] = n - 1;    // P_{i-1,j}
                        columns[5] = n;        // P_{i,j}

                        values[0] = 2.0 / pow(dx, 2) + 3.0 / pow(dy, 2);
                        values[1] = -1.0 / pow(dx, 2);
                        values[2] = -1.0 / pow(dx, 2);
                        values[3] = -1.0 / pow(dy, 2);
                        values[4] = -1.0 / dx;
                        values[5] = 1.0 / dx;

                        ierr = MatSetValues(A, 1, &m, 6, columns, values, INSERT_VALUES);
                    }
                    else // Inner Points of Domain
                    {
                        columns[0] = m;        // actual index
                        columns[1] = m - 1;    // left index
                        columns[2] = m + 1;    // right index
                        columns[3] = m - Imax; // south index
                        columns[4] = m + Imax; // north index
                        columns[5] = n;        // Pressure_{i,j}
                        columns[6] = n - 1;    // Pressure_{i-1,j}

                        values[0] = 2.0 / pow(dx, 2) + 2.0 / pow(dy, 2);
                        values[1] = -1.0 / pow(dx, 2) - Re * x_getm / (2 * dx);
                        values[2] = -1.0 / pow(dx, 2) + Re * x_getm / (2 * dx);
                        values[3] = -1.0 / pow(dy, 2) - Re * Vbar / (2 * dy);
                        values[4] = -1.0 / pow(dy, 2) + Re * Vbar / (2 * dy);
                        values[5] = 1.0 / dx;
                        values[6] = -1.0 / dx;

                        ierr = MatSetValues(A, 1, &m, 7, columns, values, INSERT_VALUES);
                    }
                }
            }
        }

        // y-momentum equation
        for (j = 0; j < Jmax; j++)
        {
            for (i = 0; i < Imax - 1; i++)
            {
                m = j * (Imax - 1) + i + nu;

                if (j == 0 || j == Jmax - 1) // Bottom and Top Edge of the Domain
                {
                    ierr = MatSetValue(A, m, m, 1.0, INSERT_VALUES);
                }
                else
                {
                    //       index of P      + number of total u & v
                    n = j * (Imax - 1) + i + nu + nv;

                    // Handling non-linear term
                    PetscInt m_Ubar = j * Imax + i; // row index of u_{i,j}^n
                    // Finding Indices for \Bar{V}
                    barInd[0] = m_Ubar;
                    barInd[1] = m_Ubar + 1;
                    barInd[2] = m_Ubar - Imax;
                    barInd[3] = m_Ubar - Imax + 1;
                    // Getting Values from x vector
                    VecGetValues(x, 4, barInd, barValues);
                    PetscScalar Ubar; // total value
                    for (int i = 0; i < 4; i++)
                    {
                        Ubar += barValues[i];
                    }
                    Ubar /= 4.0;
                    VecGetValues(x, 1, &m, &x_getm);

                    if (i == 0) // Left Edge of the Domain
                    {
                        columns[0] = m;              // actual index
                        columns[1] = m + 1;          // right index
                        columns[2] = m - (Imax - 1); // south index
                        columns[3] = m + (Imax - 1); // north index
                        columns[4] = n - (Imax - 1); // P_{i,j-1}
                        columns[5] = n;              // P_{i,j}

                        values[0] = 3.0 / pow(dx, 2) + 2.0 / pow(dy, 2) + Re * Ubar / (2 * dx);
                        values[1] = -1.0 / pow(dx, 2) + Re * Ubar / (2 * dx);
                        values[2] = -1.0 / pow(dy, 2) - Re * x_getm / (2 * dy);
                        values[3] = -1.0 / pow(dy, 2) + Re * x_getm / (2 * dy);
                        values[4] = -1.0 / dy;
                        values[5] = 1.0 / dy;

                        ierr = MatSetValues(A, 1, &m, 6, columns, values, INSERT_VALUES);
                    }
                    else if (i == Imax - 2) // Right Edge of the Domain
                    {
                        columns[0] = m;              // actual index
                        columns[1] = m - (Imax - 1); // south index
                        columns[2] = m + (Imax - 1); // north index
                        columns[3] = n - (Imax - 1); // P_{i,j-1}
                        columns[4] = n;              // P_{i,j}
                        columns[5] = m - 1;

                        values[0] = 2.0 / pow(dy, 2) - Re * Ubar / dx;
                        values[1] = -1.0 / pow(dy, 2) - Re * x_getm / (2 * dy);
                        values[2] = -1.0 / pow(dy, 2) + Re * x_getm / (2 * dy);
                        values[3] = -1.0 / dy;
                        values[4] = 1.0 / dy;
                        values[5] = Re * Ubar / dx;

                        ierr = MatSetValues(A, 1, &m, 6, columns, values, INSERT_VALUES);
                    }
                    else // Inner Points of Domain
                    {
                        columns[0] = m;              // actual index
                        columns[1] = m - 1;          // left index
                        columns[2] = m + 1;          // right index
                        columns[3] = m - (Imax - 1); // south index
                        columns[4] = m + (Imax - 1); // north index
                        columns[5] = n - (Imax - 1); // P_{i,j-1}
                        columns[6] = n;              // P_{i,j}

                        values[0] = 2.0 / pow(dx, 2) + 2.0 / pow(dy, 2);
                        values[1] = -1.0 / pow(dx, 2) - Re * Ubar / (2 * dx);
                        values[2] = -1.0 / pow(dx, 2) + Re * Ubar / (2 * dx);
                        values[3] = -1.0 / pow(dy, 2) - x_getm / (2 * dy);
                        values[4] = -1.0 / pow(dy, 2) + x_getm / (2 * dy);
                        values[5] = -1.0 / dy;
                        values[6] = 1.0 / dy;

                        ierr = MatSetValues(A, 1, &m, 7, columns, values, INSERT_VALUES);
                    }
                }
            }
        }

        // continuity equation
        for (j = 0; j < Jmax - 1; j++)
        {
            for (i = 0; i < Imax - 1; i++)
            {
                m = j * (Imax - 1) + i + nu + nv;
                ind_u = j * (Imax - 1) + i;
                ind_v = j * (Imax - 1) + i + nu;

                columns[0] = ind_u + j;        // u_{i,j}
                columns[1] = ind_u + j + 1;    // u_{i+1,j}
                columns[2] = ind_v;            // v_{i,j}
                columns[3] = ind_v + Imax - 1; // v_{i,j+1}

                values[0] = -1.0 / dx;
                values[1] = 1.0 / dx;
                values[2] = -1.0 / dy;
                values[3] = 1.0 / dy;

                ierr = MatSetValues(A, 1, &m, 4, columns, values, INSERT_VALUES);
            }
        }

        // Impose pressure coefficient at the right bottom of the matrix:
        // NOTE : not required for this problem since linear dependency
        //        does not occur due to Neumann BC.
        //ierr = MatSetValue(A, nt - 1, nt - 1, 1.0, INSERT_VALUES);

        // Fill empty diagonals in order not to get errors:
        for (int i = nu + nv; i < nt; i++)
        {
            MatSetValue(A, i, i, 1.e-10, INSERT_VALUES);
        }

        MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
        CHKERRQ(ierr);
        MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
        CHKERRQ(ierr);
        MatNorm(A, NORM_2, &norm);
        VecAssemblyBegin(RHS);
        VecAssemblyEnd(RHS);
        if (verbose)
        {
            MatView(A, PETSC_VIEWER_STDOUT_WORLD);
            CHKERRQ(ierr);
            VecView(RHS, PETSC_VIEWER_STDOUT_WORLD);
            CHKERRQ(ierr);
        }

        /*
     Create linear solver context. This will be used repeatedly for all
     the linear solves needed.
    */
        ierr = KSPCreate(PETSC_COMM_SELF, &ksp);
        CHKERRQ(ierr);
        /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix. Since all the matrices
     will have the same nonzero pattern here, we indicate this so the
     linear solvers can take advantage of this.
    */
        ierr = KSPSetOperators(ksp, A, A);
        CHKERRQ(ierr);
        KSPSetType(ksp, KSPCGLS);
        // KSPSetType(ksp, KSPPREONLY);

        /*
     Set linear solver defaults for this problem (optional).
     - Here we set it to use direct LU factorization for the solution
    */

        ierr = KSPGetPC(ksp, &pc);
        CHKERRQ(ierr);
        ierr = PCSetType(pc, PCILU);
        CHKERRQ(ierr);
        // ierr = PCFactorSetMatSolverType(pc, MATSOLVERPETSC);
        // CHKERRQ(ierr);
        // ierr = PCFactorSetUpMatSolverType(pc); /* call MatGetFactor() to create F */
        // CHKERRQ(ierr);

        // ierr = PCSetType(pc, PCJACOBI);
        // CHKERRQ(ierr);

        // ierr = KSPSetTolerances(ksp, 1.e-5, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
        // CHKERRQ(ierr);

        /*
     Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
     These options will override those specified above as long as
     KSPSetFromOptions() is called _after_ any other customization
     routines.

     Run the program with the option -help to see all the possible
     linear solver options.
    */
        ierr = KSPSetTolerances(ksp, 1.e-10, 1.e-12, PETSC_DEFAULT, PETSC_DEFAULT);
        ierr = KSPSetFromOptions(ksp);
        CHKERRQ(ierr);

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
         - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

        ierr = KSPSolve(ksp, RHS, x);
        CHKERRQ(ierr);
        /*
        View solver info; we could instead use the option -ksp_view to
        print this info to the screen at the conclusion of KSPSolve().
        */
        ierr = KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
        CHKERRQ(ierr);

        VecAssemblyBegin(x);
        VecAssemblyEnd(x);

        // ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);
        // CHKERRQ(ierr);
        // PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_PYTHON);
        viewer = PETSC_VIEWER_STDOUT_(PETSC_COMM_WORLD);
        ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_CSV);
        CHKERRQ(ierr);
        std::string _fileName = "../results/Re-" + std::to_string(Re) + "_dRe-" + std::to_string(deltaRe) + "_I-" + std::to_string(Imax) + "_J-" + std::to_string(Jmax) + ".dat";
        const char *fileName = _fileName.c_str();
        PetscViewerASCIIOpen(PETSC_COMM_WORLD, fileName, &viewer);
        VecView(x, viewer);
        PetscViewerPopFormat(viewer);

        VecGetArray(x, &x_arr);
        matplot::vector_2d u_Result;
        u_Result.resize(Jmax - 1);
        for (int j = Jmax - 2; j > -1; j--)
        {
            u_Result[j].resize(Imax);
            for (int i = 0; i < Imax; i++)
            {
                u_Result[j][i] = x_arr[j * Imax + i];
            }
        }

        if (verbose)
        {
            std::cout << "[" << std::endl;
            for (int j = 0; j < Jmax - 1; j++)
            {
                std::cout << "\t"
                          << "[ ";
                for (int i = 0; i < Imax; i++)
                {
                    std::cout << u_Result[j][i] << " ";
                }
                std::cout << "]" << std::endl;
            }
            std::cout << "]" << std::endl;

            // v vector : Results
            matplot::vector_2d v_Result;
            v_Result.resize(Jmax);
            for (int j = Jmax - 1; j > -1; j--)
            {
                v_Result[j].resize(Imax - 1);
                for (int i = 0; i < Imax - 1; i++)
                {
                    v_Result[j][i] = x_arr[j * (Imax - 1) + i + nu];
                }
            }

            std::cout << "[" << std::endl;
            for (int j = 0; j < Jmax; j++)
            {
                std::cout << "\t"
                          << "[ ";
                for (int i = 0; i < Imax - 1; i++)
                {
                    std::cout << v_Result[j][i] << " ";
                }
                std::cout << "]" << std::endl;
            }
            std::cout << "]" << std::endl;

            // p vector : Results
            matplot::vector_2d P_Result;
            P_Result.resize(Jmax - 1);
            for (int j = Jmax - 2; j > -1; j--)
            {
                P_Result[j].resize(Imax - 1);
                for (int i = 0; i < Imax - 2; i++)
                {
                    P_Result[j][i] = x_arr[j * (Imax - 1) + i + nu + nv];
                }
            }

            std::cout << "[" << std::endl;
            for (int j = 0; j < Jmax - 1; j++)
            {
                std::cout << "\t"
                          << "[ ";
                for (int i = 0; i < Imax - 1; i++)
                {
                    std::cout << P_Result[j][i] << " ";
                }
                std::cout << "]" << std::endl;
            }
            std::cout << "]" << std::endl;
        }

        Re += deltaRe;
    }

    // plot(u_Result, Imax, Jmax);
    // xxx();
    ierr = MatDestroy(&A);
    CHKERRQ(ierr);
    ierr = VecDestroy(&RHS);
    CHKERRQ(ierr);
    ierr = VecDestroy(&x);
    CHKERRQ(ierr);
    PetscFinalize();

    return ierr;
};

void plot(matplot::vector_2d &inpVec, int nx, int ny)
{
    using namespace matplot;

    vector_1d xC, yC;
    double dx = 1 / (nx - 1);
    double dy = 1 / (ny - 1);
    for (int i = 0; i <= nx; i++)
    {
        xC.push_back(i * dx + dx / 2);
    }
    for (int i = 0; i < ny; i++)
    {
        yC.push_back(i * dy + dy / 2);
    }

    auto xCoords = linspace(0.0, 5, nx);
    auto yCoords = linspace(0.0, 1.0, ny - 1);
    auto mesh = meshgrid(xCoords, yCoords);
    auto il = linspace(0.0, 1.125, 10);
    peaks();

    vector_2d X = std::get<0>(mesh);
    vector_2d Y = std::get<1>(mesh);
    // contourf(xC, yC, inpVec, vector_1d{10});
    // contourf(yCoords, xCoords, inpVec, vector_1d{10});
    contourf(X, Y, inpVec);

    show();
}