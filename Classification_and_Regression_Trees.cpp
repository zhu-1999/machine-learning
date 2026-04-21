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
#include <unordered_map>
using namespace std;
struct Node{
    bool flag;
    int index;
    double value;
    double output;
    Node *leftnode,*rightnode;
};
vector<vector<double>> read_csv(string filename){
    vector<vector<double>> data;
    ifstream file(filename);
    string line;
    while (getline(file, line))
    {
        replace(line.begin(),line.end(),',',' ');
        vector<double> a;
        stringstream ss(line);
        double num;
        while (ss>>num)
        {
            a.push_back(num);
        }
        data.push_back(a);
    }
    return data;
}
double calculate_gini(vector<vector<double>> &data){
    if(data.size()==0) return 0;
    unordered_map<double,int> count_map;
    for(int i=0;i<data.size();i++) count_map[data[i].back()]++;
    double sum_p=0;
    for(auto pair:count_map) sum_p+=pow((double)pair.second/data.size(),2);
    return 1.0-sum_p;
}
double gini(vector<vector<double>>& left, vector<vector<double>>& right) {
    double gini_left=calculate_gini(left);
    double gini_right=calculate_gini(right);
    int n=left.size()+right.size();
    return (left.size()*gini_left+right.size()*gini_right)/n;
}
void split_data(vector<vector<double>>& data,int idx,double val,vector<vector<double>>& left,vector<vector<double>>& right) {
    for(int i=0;i<data.size();i++){
        if(data[i][idx]<val) left.push_back(data[i]);
        else right.push_back(data[i]);
    }
}
Node* get_split(vector<vector<double>>& data){
    double best_gini=1e9,best_val;
    int best_index=-1;
    for(int i=0;i<data[0].size()-1;i++){
        for(int j=0;j<data.size();j++){
            vector<vector<double>> left,right;
            split_data(data,i,data[j][i],left,right);
            double g=gini(left,right);
            if(g<best_gini){
                best_gini=g;
                best_index=i;
                best_val=data[j][i];
            }
        }
    }
    Node* node=new Node();
    node->flag=false;
    node->index=best_index;
    node->value=best_val;
    node->leftnode=NULL;
    node->rightnode=NULL;
    return node;
}
double to_leaf(vector<vector<double>>& data){
    unordered_map<double,int> count_map;
    for(int i=0;i<data.size();i++) count_map[data[i].back()]++;
    double max_label;
    int max_count=0;
    for(auto pair:count_map){
        if(pair.second>max_count){
            max_label=pair.first;
            max_count=pair.second;
        }
    }
    return max_label;
}
void build(Node* node,vector<vector<double>>& data,int depth,int max_depth,int min_size){
    if(data.size()<=min_size||depth>=max_depth){
        node->flag=true;
        node->output=to_leaf(data);
        return;
    }
    vector<vector<double>> left,right;
    split_data(data,node->index,node->value,left,right);
    if(left.empty()||right.empty()){
        node->flag=true;
        node->output=to_leaf(data);
        return;
    }
    node->leftnode=get_split(left);
    build(node->leftnode,left,depth+1,max_depth,min_size);
    node->rightnode=get_split(right);
    build(node->rightnode,right,depth+1,max_depth,min_size);
}
double predict(Node* node,vector<double> row){
    if(node->flag==true) return node->output;
    else{
        if(row[node->index]<node->value) return predict(node->leftnode,row);
        else return predict(node->rightnode,row);
    }
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
vector<double> evaluate(vector<vector<double>>& dataset, int k,int max_depth,int min_size){
    vector<vector<vector<double>>> folds=k_fold_split(dataset,k);
    vector<double>scores;
    for(int i=0;i<k;i++){
        vector<vector<double>>test=folds[i],train;
        for(int j=0;j<k;j++){
            if(j!=i){
                for(int s=0;s<folds[j].size();s++) train.push_back(folds[j][s]);
            }
        }
        double score=0;
        Node* tree=get_split(train);
        build(tree,train,1,max_depth,min_size);
        vector<double> preds;
        for(int j=0;j<test.size();j++){
            if(abs(predict(tree,test[j])-test[j].back())<1e-10) score+=1;
        }
        score=score/test.size()*100;
        scores.push_back(score);
    }
    return scores;
}
int main(){
    srand(time(0)); 
    vector<vector<double>> dataset = read_csv("banknote.csv");
    vector<double> scores=evaluate(dataset,5,10,5);
    for(int i=0;i<scores.size();i++){
    cout << "Fold " << i+1 << " score: " << scores[i] <<'%'<< endl;
    }
    double sum = 0;
    for(int i=0;i<scores.size();i++) sum += scores[i];
    cout << "Average score: " << sum / scores.size() <<'%'<< endl;
    cin.get();
    cin.get();
    return 0;
}
