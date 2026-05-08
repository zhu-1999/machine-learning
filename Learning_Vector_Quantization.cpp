#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
using namespace std;
vector<vector<double>> read_csv(string filename)
{
    vector<vector<double>> data;
    ifstream file(filename);
    string line;
    while (getline(file, line))
    {
        replace(line.begin(),line.end(),',',' ');
        vector<double> row;
        stringstream ss(line);
        double num;
        while (ss>>num)
        {
            row.push_back(num);
        }
        data.push_back(row);
    }
    return data;
}
void normalize(vector<vector<double>>& dataset){
    double eps=1e-15;
    for(int j=0;j<dataset[0].size()-1;j++){
        double data_max=dataset[0][j],data_min=dataset[0][j];
        for(int i=1;i<dataset.size();i++){
            if(data_max<dataset[i][j]) data_max=dataset[i][j];
            if(data_min>dataset[i][j]) data_min=dataset[i][j];
        }
        if(data_max==data_min){
            for(int i=0;i<dataset.size();i++) dataset[i][j]=0;
        }
        else{
            for(int i=0;i<dataset.size();i++) dataset[i][j]=(dataset[i][j]-data_min)/(data_max-data_min+eps);
        }
    }
}
double distance(int dim,vector<double>&a,vector<double>&b){
    double dist=0;
    for(int i=0;i<dim;i++) dist+=(a[i]-b[i])*(a[i]-b[i]);
    return sqrt(dist);
}
int find_bmu(int dim,vector<vector<double>>&codebooks,vector<double>&x){
    double min_dist=distance(dim,codebooks[0],x),min_num=0;
    for(int i=1;i<codebooks.size();i++){
        double dist=distance(dim,codebooks[i],x);
        if(dist<min_dist){
            min_dist=dist;
            min_num=i;
        }
    }
    return min_num;
}
vector<vector<double>> init_codebooks(vector<vector<double>> train,int n_codebooks){
    vector<vector<double>> codebooks;
    shuffle(train.begin(), train.end(), mt19937(random_device{}()));
    for(int i=0;i<n_codebooks;i++) codebooks.push_back(train[i]);
    return codebooks;
}

vector<vector<double>> train_lvq(vector<vector<double>>& train,int n_codebooks,double l_rate,int epochs){
    vector<vector<double>> codebooks=init_codebooks(train,n_codebooks);
    int dim=train[0].size()-1;
    for(int i=0;i<epochs;i++){
        double cul_rate=l_rate*(1-(double)i/epochs);
        for(auto x:train){
            int x_label=(int)x.back();
            int bmu=find_bmu(dim,codebooks,x);
            int bmu_label=(int)codebooks[bmu].back();
            if(x_label==bmu_label){
                for(int j=0;j<dim;j++){
                    codebooks[bmu][j]+=cul_rate*(x[j]-codebooks[bmu][j]);
                }
            }
            else{
                for(int j=0;j<dim;j++){
                    codebooks[bmu][j]-=cul_rate*(x[j]-codebooks[bmu][j]);
                }
            }
        }
    }
    return codebooks;
}
double predict(vector<vector<double>>& codebooks,vector<vector<double>>& test){
    int correct=0,n=test.size(),dim=test[0].size()-1;
    for(int i=0;i<n;i++){
        int num=find_bmu(dim,codebooks,test[i]);
        if((int)codebooks[num].back()==(int)test[i].back()) correct++;
    }
    return (double)correct/n;
}


vector<vector<vector<double>>> k_fold_split(vector<vector<double>> dataset,int k){
    shuffle(dataset.begin(), dataset.end(), mt19937(random_device{}()));
    int n=dataset.size()/k;
    vector<vector<vector<double>>> folds(k);
    for(int i=0;i<k;i++){
        int start=i*n,end=(i+1)*n-1;
        if(i==k-1) end=dataset.size()-1;
        for(int j=start;j<=end;j++) folds[i].push_back(dataset[j]);
    }
    return folds;
}
vector<double> evaluate(vector<vector<double>>& dataset, int k,int n_codebooks,double l_rate,int epochs){
    vector<vector<vector<double>>> folds=k_fold_split(dataset,k);
    vector<double>scores;
    for(int i=0;i<k;i++){
        vector<vector<double>>test=folds[i],train;
        for(int j=0;j<k;j++){
            if(j!=i){
                for(int s=0;s<folds[j].size();s++) train.push_back(folds[j][s]);
            }
        }
        vector<vector<double>> codebooks=train_lvq(train,n_codebooks,l_rate,epochs);
        double score=predict(codebooks,test);
        scores.push_back(score);
    }
    return scores;
}

int main(){
    srand(time(0)); 
    vector<vector<double>> dataset = read_csv("ionosphere-full.csv");
    normalize(dataset);

    vector<double> scores=evaluate(dataset,5,20,0.3,50);
    for(int i=0;i<scores.size();i++){
    cout << "Fold " << i+1 << " score: " << scores[i] << endl;
    }
    double sum = 0;
    for(int i=0;i<scores.size();i++) sum += scores[i];
    cout << "Average score: " << sum / scores.size() << endl;
    cin.get();
    cin.get();
    return 0;
}
