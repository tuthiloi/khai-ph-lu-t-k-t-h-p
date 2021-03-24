/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekapr;

import java.io.BufferedWriter;
import java.io.FileWriter;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Lợii
 */
public class MyNeuralNetwordModel extends MyKnowledgeModel {
    MultilayerPerceptron neural;

    public MyNeuralNetwordModel() {
        super();
    }

    public MyNeuralNetwordModel(String filename, String m_opts, String d_opts) throws Exception {
        super(filename, m_opts, d_opts);
    }
    

    public void buildNeuralNetwork(String filename) throws Exception {
        //doc train set vao bo nho
        setTrainset(filename);
        this.trainset.setClassIndex(this.trainset.numAttributes() -1);
        //Huan luyen mo hinh neural
        this.neural=new MultilayerPerceptron();
        neural.setOptions(this.model_options);
        neural.buildClassifier(this.trainset);
    }
    
    public void evaluateNeuralNetwork(String filename) throws Exception {
        //doc train set vao bo nho
        setTestset(filename);
        this.testset.setClassIndex(this.testset.numAttributes() -1);
        //Danh gia mo hinh bang 10-fold cross-validation
        Random rnd =new Random(1);
        int folds=10;
        Evaluation eval =new Evaluation(this.trainset);
        eval.crossValidateModel(neural,this.testset, folds, rnd);
        System.out.println(eval.toSummaryString("\nKet qua danh gia mo hinh 10-fold cross-Validation\n-----\n",false));
        
    }
   public void predictClassLabel (String fileIn,String fileOut) throws Exception {
        //doc train set vao bo nho
       DataSource ds= new DataSource(fileIn);
       Instances unlabel =ds.getDataSet();
       unlabel.setClassIndex(this.testset.numAttributes() -1);
        //Du doan classLabel cho tung instance
        for(int i=0;i<unlabel.numInstances();i++){
            double predict =neural.classifyInstance(unlabel.instance(i));
            unlabel.instance(i).setClassValue(predict);
        }
        //Xuat ket qua ra fileOut
        BufferedWriter outWriter=new BufferedWriter (new FileWriter(fileOut));
       outWriter.write(unlabel.toString());
        outWriter.newLine();
        outWriter.flush();
        outWriter.close();
        
    }

    @Override
    public String toString() {
        return this.neural.toString(); //To change body of generated methods, choose Tools | Templates.
    }

    
}
