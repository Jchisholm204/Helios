/* Copyright (c) 2011-2017 Graph500 Steering Committee
   All rights reserved.
   Developed by:                Anton Korzh anton@korzh.us
                                Graph500 Steering Committee
                                http://www.graph500.org
   New code under University of Illinois/NCSA Open Source License
   see license.txt or https://opensource.org/licenses/NCSA
*/

#ifndef _MPI_AML_H_
#define _MPI_AML_H_
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif
// MPI-like init,finalize calls
extern int aml_init(int *, char ***);
extern void aml_finalize(void);
// barrier which ensures that all AM sent before the barrier are completed
// everywhere after the barrier
extern void aml_barrier(void);
// register active message function(collective call)
extern void aml_register_handler(void (*f)(int, void *, int), int n);
// send AM to another(myself is ok) node
// execution of AM might be delayed till next aml_barrier() call
extern void aml_send(void *srcaddr, int type, int length, int node);

// rank and size
extern int aml_my_pe(void);
extern int aml_n_pes(void);
extern int aml_lgn_pes(void);
extern int aml_node_size(void);
extern int aml_node_id(void);

#ifdef __cplusplus
}
#endif

#define my_pe aml_my_pe
#define num_pes aml_n_pes

// Number of running processes
#define n_procs aml_n_pes()
// ID of current process
#define proc_id aml_my_pe()
// Log2 of number of processes
#define lgprocs aml_lgn_pes()

// Number of running processes on the current node
#define node_size aml_node_size()
// ID of process on node
#define node_id aml_node_id()

extern MPI_Comm comm, comm_intra;

#define aml_time() MPI_Wtime()
#define aml_long_allsum(p)                                                     \
    MPI_Allreduce(MPI_IN_PLACE, p, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD)
#define aml_long_allmin(p)                                                     \
    MPI_Allreduce(MPI_IN_PLACE, p, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD)
#define aml_long_allmax(p)                                                     \
    MPI_Allreduce(MPI_IN_PLACE, p, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD)

#endif
