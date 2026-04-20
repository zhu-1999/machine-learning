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
vector<double> train_sgd(vector<vector<double>>& train,double learning_rate,int epochs){
    int n=train[0].size();
    vector<double>b(n,0);
    for(int i=0;i<epochs;i++){
        for(int j=0;j<train.size();j++){
            double y_true=train[j][n-1],y_pred=b[0],error;
            for(int k=0;k<n-1;k++) y_pred+=b[k+1]*train[j][k];
            error=y_pred-y_true;
            b[0]-=learning_rate*error;
            for(int k=0;k<n-1;k++) b[k+1]-=learning_rate*error*train[j][k];
        }
    }
    return b;
}
double RMSE(vector<vector<double>>& test, vector<double>& b){
    double rmse=0;
    for(int i=0;i<test.size();i++){
        int n=b.size();
        double y_pred=b[0];
        for(int j=0;j<n-1;j++) y_pred+=b[j+1]*test[i][j];
        rmse+=pow(y_pred-test[i][n-1],2);
    }
    return sqrt(rmse/test.size());
}
vector<vector<vector<double>>> k_fold_split(vector<vector<double>> dataset, int k){
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
vector<double> evaluate(vector<vector<double>>& dataset, int k,double learning_rate,int epochs){
    vector<vector<vector<double>>> folds=k_fold_split(dataset,k);
    vector<double>scores;
    for(int i=0;i<k;i++){
        vector<vector<double>>test=folds[i],train;
        for(int j=0;j<k;j++){
            if(j!=i){
                for(int s=0;s<folds[j].size();s++) train.push_back(folds[j][s]);
            }
        }
        vector<double> b=train_sgd(train,learning_rate,epochs);
        double score=RMSE(test,b);
        scores.push_back(score);
    }
    return scores;
}
int main(){
    srand(time(0)); 
    vector<vector<double>> dataset = read_csv("winequality-white.csv");
    normalize(dataset);
    vector<double> scores=evaluate(dataset,10,0.001,50);
    for(int i=0;i<scores.size();i++){
    cout << "Fold " << i+1 << " RMSE: " << scores[i] << endl;
    }
    double sum = 0;
    for(int i=0;i<scores.size();i++) sum += scores[i];
    cout << "Average RMSE: " << sum / scores.size() << endl;
    cin.get();
    cin.get();
    return 0;
}