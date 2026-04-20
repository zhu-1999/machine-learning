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
double mean(vector<double>& value){
    double sum=0;
    for(int i=0;i<value.size();i++)
        sum+=value[i];
    return sum/value.size();
}
double variance(vector<double>& value,double mean_value){
    double var=0;
    for(int i=0;i<value.size();i++)
        var+=(value[i]-mean_value)*(value[i]-mean_value);
    return var;
}
double covariance(vector<double>& a,vector<double>& b,double mean_a,double mean_b){
    double cov=0;
    for(int i=0;i<a.size();i++)
        cov+=(a[i]-mean_a)*(b[i]-mean_b);
    return cov;
}
vector<double> coefficients(vector<vector<double>>& dataset){
    vector<double>x,y;
    for(int i=0;i<dataset.size();i++){
        x.push_back(dataset[i][0]);
        y.push_back(dataset[i][1]);
    }
    double mean_x,mean_y,cov,var_x,b0,b1;
    mean_x=mean(x);
    mean_y=mean(y);
    cov=covariance(x,y,mean_x,mean_y);
    var_x=variance(x,mean_x);
    b1=cov/var_x;
    b0=mean_y-b1*mean_x;
    return {b0,b1};
}

double RMSE(vector<double>& y_true, vector<double>& y_pred){
    double rmse=0;
    for(int i=0;i<y_true.size();i++)
        rmse+=(y_pred[i]-y_true[i])*(y_pred[i]-y_true[i]);
    return sqrt(rmse/y_true.size());
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
vector<double> evaluate(vector<vector<double>>& dataset, int k){
    vector<vector<vector<double>>> folds=k_fold_split(dataset,k);
    vector<double>scores;
    for(int i=0;i<k;i++){
        vector<vector<double>>test=folds[i],train;
        for(int j=0;j<k;j++){
            if(j!=i){
                for(int s=0;s<folds[j].size();s++) train.push_back(folds[j][s]);
            }
        }
        vector<double> coeff=coefficients(train),y_true,y_pred;
        double b0=coeff[0],b1=coeff[1];
        for(int j=0;j<test.size();j++) {
            y_true.push_back(test[j][1]);
            y_pred.push_back(b1*test[j][0]+b0);
        }
        double score=RMSE(y_true,y_pred);
        scores.push_back(score);
    }
    return scores;
}
int main(){
    srand(time(0)); 
    vector<vector<double>> dataset = read_csv("insurance.csv");
    vector<double> scores=evaluate(dataset,10);
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
