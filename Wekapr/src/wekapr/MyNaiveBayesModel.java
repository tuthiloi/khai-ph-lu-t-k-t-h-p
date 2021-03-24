/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekapr;

import weka.classifiers.bayes.NaiveBayes;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;


/**
 *
 * @author Lợii
 */
public class MyNaiveBayesModel extends MyKnowledgeModel {
    NaiveBayes nbayes;
    
    public MyNaiveBayesModel(){
        super();
    }

    public MyNaiveBayesModel(String filename,String m_opts,String d_opts) throws Exception {
        
        super(filename,m_opts,d_opts);
    }

   
        
    
    
    public void builNaiveBayes(String filename) throws Exception{
       //doc train set vao bo nho
        setTrainset(filename);
        this.trainset.setClassIndex(this.trainset.numAttributes() -1);
        //Huan luyen mo hinh Naivebayes
        this.nbayes=new NaiveBayes();
        //nbayes.setOption(this.model_options);
        nbayes.buildClassifier(this.trainset);
      }
     public void evaluateNaivebayes(String filename) throws Exception  {
          //doc test set vao bo nho
        setTestset(filename);
        this.testset.setClassIndex(this.testset.numAttributes() -1);
        //Danh gia mo hinh bang 10-fold bross-validation
        Random rnd =new Debug.Random(1);
        int folds =10;
        Evaluation eval =new Evaluation(this.trainset);
        eval.crossValidateModel(nbayes,this.testset,folds,rnd);
        System.out.println(eval.toSummaryString("\nKet qua danh gia mo hinh 10-fold cross-Validation\n-----\n",false));
        
     }  
      public void predictClassLabel(String fileln,String fileOut) throws Exception{
          //Doc du lieu can du lieu vao bo nho
          DataSource ds=new DataSource(fileln);
          Instances unlabel =ds.getDataSet();
          unlabel.setClassIndex(unlabel.numAttributes() -1);
          //Du doan classLabel cho tung instance
          for(int i=0;i<unlabel.numAttributes();i++){
              double predict=nbayes.classifyInstance(unlabel.instance(i));
              unlabel.instance(i).setClassValue(predict);
          }
          //Xuat ket qua ra fileOut
          BufferedWriter outWiter =new BufferedWriter(new FileWriter(fileOut));
          outWiter.write(unlabel.toString());
          outWiter.newLine();
          outWiter.flush();
          outWiter.close();
          
      }

    @Override
    public String toString() {
        return this.nbayes.toString(); //To change body of generated methods, choose Tools | Templates.
    }

 
}
