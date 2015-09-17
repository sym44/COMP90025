  //compile with -D, e.g
  //
  // gcc -fopenmp -o manbrot mandlebrot.c -DDYNAMIC
  //
  //to get the version that uses dynamic scheduling

  #include <omp.h>
  #include <mpi.h>
  #include <complex.h>

  #include <time.h>
  float timediff(struct timespec t1, struct timespec t2)
  {  if (t1.tv_nsec > t2.tv_nsec) {
       t2.tv_sec -= 1;
       t2.tv_nsec += 1000000000;
     }
     return t2.tv_sec-t1.tv_sec + 0.000000001 * (t2.tv_nsec-t1.tv_nsec);
  }


  //finds chunk among 0,....,n-1 to assign to thread number me among nth
  //threads
  void findmyrange(int n, int nth, int me, int *myrange)
  { int chunksize = n / nth;
    myrange[0] = me * chunksize;
    if (me < nth-1) myrange[1] = (me+1) * chunksize -1;
    else myrange[1] = n - 1;
  }

  #include <stdlib.h>
  #include <stdio.h>
  //returns a random permutation of n from start..end  
  int *rpermute(int n, int start) {
    int *a = (int *) (int *) malloc(n*sizeof(int));
    int k;
    int i = start;
    for (k=0; k < n; k++) a[k] = i++;
    for (k=n-1; k > 0; k--) {
      int j = rand() % (k+1);
      int temp = a[j];
      a[j] = a[k];
      a[k] = temp;
    }
    return a;
  }

  #define MAXITERS 1000

  //globals
  int count = 0, tot_count;
  int nptsside, mpi_chunksize;
  int print_node;
  float side2;
  float side4;
  int nnodes;
  int my_rank;
  int myrange[2];

  int inset(double complex c) {
    int iters;
    float rl,im;
    double complex z = c;
    for (iters = 0; iters < MAXITERS; iters++) {
      z = z*z +c;
      rl = creal(z);
      im = cimag(z);
      if (rl*rl + im*im > 4) return 0;

    }
    return 1;
  }


  int *scram;

  void dowork()
  {

    int mi;
    if (my_rank == print_node) {
      #ifdef RC
      #pragma omp parallel 
      {
        #pragma omp master 
        { printf("Using %d threads\n", omp_get_num_threads()); }
      }
      #endif

        printf("My rank is %d has %d:", my_rank, mpi_chunksize);
        if (mpi_chunksize < 20) 
          for (mi = 0; mi < mpi_chunksize; mi++) printf("%d ", scram[mi]);
        printf("\n");
    }

      int x,y; float xv, yv;
      int i;
      double complex z;

    #ifdef RC
    #pragma omp parallel for reduction(+:count) private(x,y,xv,yv,i,z)
    #else
    
      //#pragma omp single
      //{ printf("Using %d threads\n", omp_get_num_threads()); }

      #ifdef STATIC
      #pragma omp parallel for reduction(+:count) schedule(static) private(x,y,xv,yv,i,z) 
      #elif defined DYNAMIC
      #pragma omp parallel for reduction(+:count) schedule(dynamic) private(x,y,xv,yv,i,z) 
      #elif defined GUIDED
      #pragma omp parallel for reduction(+:count) schedule(guided) private(x,y,xv,yv,i,z) 
      #endif

    #endif

      for (i=0; i < mpi_chunksize; i++) {
        x = scram[i];

         for (y=0; y < nptsside; y++) {
           xv = (x - side2) / side4;
  	 yv = (y - side2) / side4;
  	 z = xv + yv*I;
  	 if (inset(z)) {
  	   count++;
  	 }
         }
      }
    

    MPI_Reduce(&count, &tot_count, 1, MPI_INT, MPI_SUM, print_node, MPI_COMM_WORLD);
  }

  int main(int argc, char **argv)
  {
    nptsside = atoi(argv[1]);
    print_node = atoi(argv[2]);
    side2 = nptsside / 2.0;
    side4 = nptsside / 4.0;


    int provided, claimed;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided );
    MPI_Query_thread( &claimed );

  //  MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nnodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == print_node) {
      printf( "Query thread level= %d  Init_thread level= %d\n", claimed, provided );
      printf( "Defined LEVEL= %d  (ompi_info | grep -i thread) \n", MPI_THREAD_MULTIPLE);
    }
  	        

    mpi_chunksize = nptsside/nnodes;

    struct timespec bgn,nd;
    clock_gettime(CLOCK_REALTIME, &bgn);

    #ifdef RC
    scram = rpermute(nptsside, 0);
    MPI_Scatter(scram, mpi_chunksize, MPI_INT, scram, mpi_chunksize, MPI_INT, 0, MPI_COMM_WORLD);
    #else

      findmyrange(nptsside, nnodes, my_rank, myrange);
      scram = rpermute(mpi_chunksize, myrange[0]);
      printf("My range is %d %d \n", myrange[0], myrange[1]);
    #endif

    
    dowork();

    //implied barrier

    clock_gettime(CLOCK_REALTIME, &nd);

    if (my_rank == print_node) {
      printf("%d\n", tot_count);
      printf("Random chunk RC is defined\n");
      printf("%f\n", timediff(bgn,nd));
    }

    MPI_Finalize();

  }
