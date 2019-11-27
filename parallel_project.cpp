#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include<bits/stdc++.h>

#define MATRIX_SIZE 2048
#define SMALL_SIZE 3

using namespace std;


class Timer {

public:
    Timer(const char* header ="")
        : beg_(clock_::now()), header(header) {}
    ~Timer() {
            double e = elapsed();
            cout << header << ": " << e << " micros" << endl;
    }
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1,1000000> >
                second_;
    std::chrono::time_point<clock_> beg_;
    const char* header;

};

class GaussElimination {

public:
    static void serial(vector<double> M, vector<double> b, const int size) {
	    
	    vector<double> x(size);

	    {
	    	
	    	Timer t("\n\nSERIAL\t\t");
	        for(int i = 0; i < size - 1; i++) {
	            for(int j = i + 1; j < size; j++) {
	                double k = -M[j * size + i] / M[i * size + i];
	                b[j] += b[i] * k;

	                for(int l = i; l < size; l++) {
	                    M[j * size + l] += M[i * size + l] * k;
	                }
	            }
	        }

	        x[size - 1] = b[size - 1] / M[size * size - 1];
	        for(int i = size - 2; i >= 0; i--) {
	            double sum = 0;
	            for(int j = i + 1; j < size; j++) {
	                sum += M[i * size + j] * x[j];
	            }
	            x[i] = (b[i] - sum) / M[i * size + i];
	        }
	        
	    }
	    cout<<"AUGMENTED MATRIX AFTER ROW OPERATIONS:\n";

	    int size2 = min(size,7);
	    cout<<"\n";
	    for(int i=0;i<size2;i++){
	    	cout<<"\t";
	    	for(int j=0;j<size2;j++){

	    		cout<<M[i*size + j]<<"\t";

	    	}

	    	cout<<"\t|\t"<<b[i]<<"\t\t"<<endl;
	    }
	    cout<<"\n\nSOLUTION VECTOR: \n";
	    for(int i=0;i<size2;i++){
	    	cout<<"\tx"<<i+1<<":\t"<<x[i]<<endl;
	    }
        return;
    }

public:
    static void parallel(vector<double> M, vector<double> b, const int size) {
    	vector<double> x(size);
	    {
	    	Timer t("\n\nPARALLEL\t");
	        double k;
	        int i, j, l;

	        for(i = 0; i < size - 1; i++) {
	            #pragma omp parallel for shared(M, b) private(k, j, l)
	            for(j = i + 1; j < size; j++) {
	                double k = -M[j * size + i] / M[i * size + i];
	                b[j] += b[i] * k;

	                #pragma omp simd
	                for(l = i; l < size; l++) {
	                    M[j * size + l] += M[i * size + l] * k;
	                }
	            }
	        }
	        
	        
	    }
	    x[size - 1] = b[size - 1] / M[size * size - 1];
        for(int i = size - 2; i >= 0; i--) {
            double sum = 0;

            #pragma omp simd
            for(int j = i + 1; j < size; j++) {
                sum += M[i * size + j] * x[j];
            }
            x[i] = (b[i] - sum) / M[i * size + i];
        }


	    cout<<"AUGMENTED MATRIX AFTER ROW OPERATIONS:\n";

	    int size2 = min(size,7);
	    cout<<"\n";
	    for(int i=0;i<size2;i++){
	    	cout<<"\t";
	    	for(int j=0;j<size2;j++){

	    		cout<<M[i*size + j]<<"\t";

	    	}

	    	cout<<"\t|\t"<<b[i]<<"\t\t"<<endl;
	    }
	    cout<<"\n\nSOLUTION VECTOR: \n";
	    for(int i=0;i<size2;i++){
	    	cout<<"\tx"<<i+1<<":\t"<<x[i]<<endl;
	    }
    }

};

int main()
{
	int size = SMALL_SIZE;
	cout<<"We will solve the set of linear equations: Ax = B using GAUSSIAN ELIMINATION";

	cout<<"\n\nEnter the dimension (n X n): ";
	cin>>size;
	cout<<endl;

  //  for(int j = 0; j < 1; j++) {

        vector<double> M(size*size);
        if(size<10) cout<<"Input the A Matrix:\n";
        for(int i = 0; i < size * size; i++) {
        	
        	if(size>=10) {
        		M[i] = rand() ;//% 100 - 50;
        	}

        	else cin>>M[i];
        }
        vector<double> b(size);
        for(int i = 0; i < size; i++) {
        	
        	if(size>=10) {
        		b[i] = rand() ;//% 100 - 50;
        	}

        	else cin>>b[i];
        }

		GaussElimination::parallel(M, b, size);
        GaussElimination::serial(M, b, size);

   // }

    return 0;
}