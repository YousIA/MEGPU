#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <time.h>
//#include <cutil.h> 

using namespace std;
# define r 40
# define M 1000   // number of items
# define N  90 // number of transactions
# define alpha 1 // represents the weight of the support in the first fitness function
# define Beta 1 // represents the weight of the confidence in the first fitness function
# define k 15   // number of bees
struct ligne {int trans[N]; int nb;} *lg;
struct bee {int solution[N]; float cost;} *be;
/**************prototype declaration*******/
void read_trans(ligne T[]);// this function allows to read the transactional data base et insert it into the dataset vector
void display_dataset(ligne T[]); //this function allows to display the transactional data base
void display_solution(bee S); // this function display the current solution with its cost
float support_rule(ligne T[], int s[]); // this function calculates the support of the entire solution s
float support_antecedent(ligne T[], int s[]); // this function computes the support of the antecedent of the solution s
float confidence(int sr, int sa); // it calculates the confidence of the rule
float fitness1(int sr, int sa); // computes the fitness of a given solution s
void create_Sref(bee *s, ligne V[]); // here we create the solution reference sref and initialize it with the random way
bee neighborhood_computation(bee S, bee *V, ligne *D);// this function explores the local region for each bee
void search_area1(bee s, bee *T, int iteration, ligne V [],int flip); //detremines the search area for each bee using the first strategy
void search_area2(bee s, bee *T, int iteration, ligne V[], int flip); //detremines the search area for each bee using the second strategy
void search_area3(bee s, bee *T, int iteration, ligne V[], int distance); //detremines the search area for each bee using the third strategy
int W(int t[]); // indicates the  weight of solution representing by a vector t, this function is used on search_area3()
void copy(int t[], int v[]); // it copies the vector t in the vector v
int best_dance(bee *T); // return the best dance after the exploration of search region of each bee
void parallel_fitness(bee *V, ligne *D); // parallelize solution computing 
void display_bees(bee T[]); // display solutions
/*************************************************************************************/
__global__ void KernelSupport_rules(bee *N_List_GPU, int **compt_GPU, struct ligne *dataset_GPU){
	int thread_idx ;
	thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	bool appartient=true;
    int indice=blockIdx.x*1000;
	indice=thread_idx-indice;

		int j=1;
		while (j<N){
			if (N_List_GPU[blockIdx.x].solution[indice]!=0){
				int l=0; 
				bool existe=false;
				while (l< dataset_GPU[thread_idx].nb && existe==false){
					if (dataset_GPU[thread_idx].trans[l]==j){
						existe=true;
					}
					l++;
				}
				if (existe==false){
					appartient=false;
				}
			}
			j++;    
		}
		if (appartient==true){
			//compt_GPU[blockIdx.x][thread_idx]=1;
		}
	
   // }
}

__global__ void KernelSupport_antecedent(bee *N_List_GPU, int **compt_GPU, struct ligne *dataset_GPU){
	int thread_idx ;
	thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	bool appartient=true;
    int indice=blockIdx.x*M;
	indice=thread_idx-indice;
		int j=1;
		while (j<N){
			if (N_List_GPU[blockIdx.x].solution[indice]==1){
				int l=0; 
				bool existe=false;
				while (l< dataset_GPU[thread_idx].nb && existe==false){
					if (dataset_GPU[thread_idx].trans[l]==j){
						existe=true;
					}
					l++;
				}
				if (existe==false){
					appartient=false;
				}
			}
			j++;    
		}
		if (appartient==true){
			//compt_GPU[blockIdx.x][thread_idx]=1;
		}
	  
       //}
}
int main(void){
    FILE *f=NULL;
    f=fopen("/home/ydjenouri/mesprog/resultat1.txt","a");
    struct ligne *dataset_CPU, *dataset_GPU;
    struct bee *T_Dance;
	struct bee *N_List_CPU;
    struct bee Sref;
    struct bee best;
    int flip=1, distance, IMAX=1;
   // int k=5;
    cudaEvent_t start, stop;
    float  elapsedTime;
    int j;
    /*****************************parallel program***********************/
	//allocation de la memoire dans le CPU
	dataset_CPU = (ligne *) malloc(M * sizeof(ligne)) ;
	T_Dance = (bee *) malloc(k * sizeof(bee)) ;
	N_List_CPU=(bee *) malloc(k *sizeof(bee));
       ////allocation de la memoire dans le GPU
       cudaMalloc( (void**) &dataset_GPU, M*sizeof(ligne));
       
	   //read transactional database and insert in the dataset_CPU
       read_trans(dataset_CPU);
	   cudaMemcpy(dataset_GPU, dataset_CPU, M * sizeof(ligne), cudaMemcpyHostToDevice);
       cudaEventCreate( &start );
       cudaEventCreate( &stop );
       cudaEventRecord( start, 0 ) ;
       create_Sref(&Sref, dataset_GPU); // creer une solution reference
	//display_solution(Sref);
      search_area1(Sref, T_Dance, IMAX, dataset_GPU, flip);
      printf("hello");
     // display_bees(T_Dance);
     // for ( int k=5; k<=15;k=k+5)
     for ( int i=0; i<=IMAX;i++)
	{
	    			
	    for ( j=0;j<k;j++) // neighborhood computation for all the solution in tab
					{ 
				       T_Dance[j]=neighborhood_computation(T_Dance[j], N_List_CPU, dataset_GPU);
					}
					/*j=best_dance(T_Dance,k);
					copy(T_Dance[j].solution,Sref.solution);
					Sref.cost=T_Dance[j].cost;
					if (Sref.cost > best.cost)//atte o maximisation
			    { 
					 copy(Sref.solution, best.solution);
					 best.cost=Sref.cost;
			    }
				*/
		       //display_bees(T_Dance);	 
//			//average=best.cost+average; 	        
			//printf("\nk="+b.k+" IMAX="+b.IMAX+"  average fitness="+average);
             search_area1(Sref,T_Dance, i, dataset_GPU, flip);
	
       } //Bso ending

    cudaEventRecord( stop, 0 ) ;
    cudaEventSynchronize( stop ) ;
    cudaEventElapsedTime( &elapsedTime,start, stop ) ;
    printf("K=%d IMAX=%d  Execution Time in GPU : %3.1f ms\n", k,IMAX, elapsedTime );
   // fprintf(f,"K=%d IMAX=%d flip=%d Execution Time in GPU : %3.1f ms\n", k,flip,IMAX, elapsedTime );
    printf("Yes\n");
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
//}// end loop IMAX
//} // end loop flip
//} // end loop k

//fclose(f);
cudaFree(dataset_GPU);

return 0;
}
/**********************copry t in v********/
void copy(int t[], int v[])
{
for (int i=0;i<N; i++)
{
v[i]=t[i]; 
}     
}
/*******read transactional data bass and insert it in the data set structure********************************/
void read_trans(ligne T[]){
	char c='4';
	char t[100];
	int j;
	int i=0;
	int l=0;
	FILE *f=NULL;
	f=fopen("/home/ydjenouri/mesprog/T_90_1000.txt","r");
	if (f!=NULL) {
		//cout<<"the file is succefully opened"<<endl;
		j=0;
		while (c!=EOF){
			c=fgetc(f);
			if (c==' '){
				t[j]='\0';
				T[i].trans[l]=atoi(t);
                            l++;
				j=0;
			}
			if (c=='\n'){
				T[i].nb=l;
				l=0;
				i++;
				j=0;
			}
			if (c!=' ' && c!='\n'){
				t[j]=c;
				j++;
			}
		}   
		fclose(f);
	}
}
/*************************compute the support of the solution s**********/
float support_rule(ligne T[], int s[])
{
float compt=0;
		for (int i=0; i<M; i++)
		{
		bool appartient=true;
		
		int j=1;
		while (j<N)
		{
		 if (s[j]!=0)
		{
			int l=0; 
			bool existe=false;
			while (l< T[i].nb && existe==false)
			{
				if (T[i].trans[l]==j)
				{existe=true;}
			l++;
			}
			if (existe==false){appartient=false;}
		}
		j++;	
		}
		if (appartient==true) {compt++;}
		}
	   compt=compt/M;
	return compt;
}
/*****************************support antecedent computing*****************************/
float support_antecedent(ligne T[], int s[])
	{
             float compt=0;
		
		for (int i=0; i<M; i++)
		{
		bool appartient=true;
		int j=1;
		while (j<N)
		{
		 if (s[j]==1 ||s[j]==2)
		{
			int l=0; 
			bool existe=false;
			while (l< T[i].nb && existe==false)
			{
					if (T[i].trans[l]==j)
				        {existe=true;}
			l++;
			}
			if (existe==false){appartient=false;}
		}
		j++;	
		}
		if (appartient==true) {compt++;}
		}
	   compt=compt/M;
	//if(compt!=0)System.out.println("antecedent"+compt);
	   return compt;
	}
/****************************condifence computing**************************/
float confidence(int sr, int sa)
{
	float conf=1;
	conf=(float)sr/sa;
return conf;
}
/***********************evaluation of the solution s******/
float fitness1(int sr, int sa)
	{
	float cost=0; 
	//if (support_rule(sol)<Minsup || confidence(sol)<Minconf){cout=-1;}
	float x=(float)alpha*(sr/M);
	float y=(float)Beta*confidence(sr,sa);
	cost=x+y;
	return cost;
	}
/**************************display_solution*****************/
void display_solution(bee S)
{
for (int i=0;i<N;i++)
{
    printf("%d ", S.solution[i]);
}
printf ("cost is:%f",S.cost);
printf("\n");
}

/*********************create a solution reference Sref******************************************/
void create_Sref(bee *s, ligne V[])
{
	for (int i=0;i<N;i++){
		if (rand() % 2==0){
			(*s).solution[i]=0 ;
		}
		else {
			if (rand() % 2==0){
				(*s).solution[i]=0;
			}
			else {
				(*s).solution[i]=rand() % 3; 
			}
		}
	}
       //parallel_fitness(s, V);
}
/***********************************negihborhood computation************************/
bee neighborhood_computation(bee S, bee *V, ligne *D)
{
bee s;
int indice=0;
int i=0; 
bee neighbor, best_neighbor;
float best_cost=0;
		//copy(S.solution,best_neighbor);
		   copy(S.solution,neighbor.solution);
            while (i<k)
		   {
	          	    
	          if (neighbor.solution[indice]==0) 
	          {
	        		  if (rand()%2==0)
	        		  {neighbor.solution[indice]=1;}
	        		  else{neighbor.solution[indice]=2;}
	          }
	          else {
	          if (neighbor.solution[indice]==1) 
	          {
	        	  if (rand()%2==0)
                         neighbor.solution[indice]=0;

	        	  else {
	        	 neighbor.solution[indice]=2;
	        		  }
	          }
	          else {
	          if (neighbor.solution[indice]==2) 
	          {
	        	  	  if (rand()%2==0)
                                neighbor.solution[indice]=0;

	        	  else {
	                neighbor.solution[indice]=1;
	        		 }
	          }
	          }
	          }
	     indice++;
	     if (indice>=N){indice=0;}   
	     copy(neighbor.solution,V[i].solution);
         i++;
         
		 /*if (neighbor.cost>best_cost){copy(neighbor.solution,best_neighbor.solution);
                                            best_cost=neighbor.cost;}*/
		 }
		 parallel_fitness(V, D); 
		 
//copy(best_neighbor.solution, s.solution);
//s.cost=best_cost;
s.cost=0;
return s;
}
/************************determination of search area********************/
void search_area1(bee s, bee *T, int iteration, ligne V[],int flip)
{
	 
	    int indice=iteration % N;
	    int i=0;
		   while (i<k)
		   {
			   for (int j=0;j<N;j++)
			   {   
			    T[i].solution[j]=s.solution[j];	    
			   }	
	                 if (T[i].solution[indice]==0) 
	                {
	        	         if (iteration%4==0)
	        		  {T[i].solution[indice]=1;}
	        		  else{T[i].solution[indice]=2;}
	        		  
	        	    //           }    
	                }
			
	          else{
	          if (T[i].solution[indice]==1) 
	          {	  if (iteration%3==0)
			  {T[i].solution[indice]=0;}
			  else{T[i].solution[indice]=2;}
	        		 
	        }
	          else{ 
	          
	        	  if (iteration%2==0)
	    		  {
                       T[i].solution[indice]=1;}
	    		  else{
                         T[i].solution[indice]=0;}
	        	 }
	          }
	     indice=indice+flip;
	     if (indice>=N){indice=0;}   
		 parallel_fitness(&T[i], V);
		//T_Dance[i].cost=fitness1(T_Dance[i].solution);//evaluer solution  
		 i++;
		   }
}
/**************search 2*********************/
void search_area2(bee s, bee *T, int iteration, ligne V[], int flip)
{
int i=0;
int Nb_sol=0;
bool stop=false;
	  while (i<N && stop==false)
	  {
		   for (int j=0;j<N;j++)
		   {   
			   T[Nb_sol].solution[j]=s.solution[j];	       
		   }
		   for (int l=i;l<(i+flip)%N;l++)
		   {
		  if ( T[Nb_sol].solution[l]==0) 
	     {
	   	  if (rand()%2==1)
	   		  { T[Nb_sol].solution[l]=1;}
	   		  else{T[Nb_sol].solution[l]=2;}
	   		  
	   	}
	     else {
	     if (T[Nb_sol].solution[l]==1) 
	     {
	    	 if (rand()%2==1)
	  		  {T[Nb_sol].solution[l]=0;}
	  		  else{T[Nb_sol].solution[l]=2;}
	     }
	     else {
	       if (T[Nb_sol].solution[l]==2) 
	        {
	    	 if (rand()%2==0)
	  		  {T[Nb_sol].solution[l]=0;}
	  		  else{T[Nb_sol].solution[l]=1;}
	         }
	         }
	     }
		}
        parallel_fitness(&T[i], V);
	//T_Dance[Nb_sol].cost=fitness1(T_Dance[Nb_sol].solution); //evaluates the solution  
	Nb_sol++; 
	if (Nb_sol==k){stop=true;}   
	}
}   
/********search3***************************/
int W(int t[])
{
int w=0;
	for (int i=0;i<N; i++)
	{
	w=w+t[i];
	}
return w;
} 
/*******search 3 continued****************************/
void search_area3(bee s, bee *T, int iteration, ligne V[], int distance)
{
int Nb_sol=0;
	  while (Nb_sol!=k)
	  {
		   for (int j=0;j<N;j++)
		   {   
			  T[Nb_sol].solution[j]=s.solution[j];	 	    
		   }
		   int l=0;
		   int cpt=0;
		   while (cpt<distance)
		   {
		  if (T[Nb_sol].solution[l]==0) 
	     {
	   	  if (rand()%2==1)
	   		  {T[Nb_sol].solution[l]=1; cpt++;}
	   		  else{T[Nb_sol].solution[l]=2;cpt=cpt+2;}
	   		  
	   	}
	     else {
	     if (T[Nb_sol].solution[l]==1) 
	     {
	    	 if (rand()%2==0)
	  		  {T[Nb_sol].solution[l]=0;cpt++;}
	  		  else{T[Nb_sol].solution[l]=2;cpt++;}
	     }
	     else {
	       if (T[Nb_sol].solution[l]==2) 
	        {
	    	 if (rand()%2==0)
	  		  {T[Nb_sol].solution[l]=0;cpt=cpt+2;}
	  		  else{T[Nb_sol].solution[l]=1;cpt=cpt+1;}
	         }
	         }
	     
		   }
		  l=(l+1)%N;
		   } //end the small while
         //parallel_fitness(&T[Nb_sol], V);
	//T_Dance[Nb_sol].cost=fitness1(T_Dance[Nb_sol].solution);//assecees the solution  
	Nb_sol++; 
	  } // end the big while
 }
/********************************best dance********************/
int best_dance(bee *T)
{
	float max=T[0].cost;
	int indice=0;
	for (int i=1;i<k;i++)	
	{
     	if (T[i].cost>max)
	     {     
           max=T[i].cost;
		   indice=i;
         }
	}
return indice;
}
/***********************paralelize solution computing*******/
void parallel_fitness(bee *N_List_CPU, ligne V[])
{
bee *N_List_GPU;
	//int **compt;
    //compt = (int **) malloc(k*M*sizeof(int));
       	
		/*for (int i=0;i<k;i++){
		  for (int j=0;i<M;j++){
		compt[i][j]=0;
	}
	}*/
       int **compt_GPU;
	// cudaEventCreate( &start );
     	// cudaEventCreate( &stop );
     	// cudaEventRecord( start, 0 ) ;

	cudaMalloc((void**) &N_List_GPU, k*sizeof(bee));
	//cudaMalloc( (void**) &compt_GPU, k*M* sizeof(int));
	cudaMemcpy(N_List_GPU, N_List_CPU, k *sizeof(bee),cudaMemcpyHostToDevice);
	//cudaMemcpy(compt_GPU, compt, k*M *sizeof(int),cudaMemcpyHostToDevice);

	KernelSupport_rules<<<20*N,M>>>(N_List_GPU, compt_GPU, V);
       //cudaMemcpy(compt, compt_GPU, k*M*sizeof(int),cudaMemcpyDeviceToHost);
       /*int sr=0; 
       for (int i=0;i<M;i++){
        	sr=sr+compt[i];
       }
       KernelSupport_antecedent<<<20*N,M>>>(s_GPU, compt_GPU, V);
       cudaMemcpy(compt, compt_GPU, M*sizeof(int),cudaMemcpyDeviceToHost);
       int sa=0; 
       for (int i=0;i<M;i++){
        	sa=sa+compt[i];
       }
       (*sol).cost=fitness1(sr,sa);*/
	   
}
/*****************************display T_dance************/
void display_bees(bee T[])
{
//FILE *f=NULL;
//f=fopen("/home/ydjenouri/mesprog/resultat1.txt","a");
//if (f!=NULL) {
for (int i=0;i<k;i++)
{
    for (int j=0;j<N;j++)
    {
    printf ("%d ",T[i].solution[j]);   
    }
    printf("%f", T[i].cost);
    printf("\n");
}
//fclose(f);
//}
}
