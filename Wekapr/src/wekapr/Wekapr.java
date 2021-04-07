/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekapr;

import weka.classifiers.trees.J48;

/**
 *
 * @author Lợii
 */
public class Wekapr {

    /**
     * @param args the command line arguments
     * @throws java.lang.Exception
     */
    public static void main(String[] args) throws Exception {

//         TODO code application logic here
//        MyKnowledgeModel model = new MyKnowledgeModel("D:\\weka\\Weka-3-8-5\\data\\iris.arff");
//        System.out.println(model);
//        model.saveData("D:\\data\\iris.arff");
//        model.saveData2CSV("D:\\data\\iris.csv");
//       MyAprioriModel model = new MyAprioriModel("D:\\weka\\Weka-3-8-5\\data\\weather.numeric.arff","-N 10 -T 0 -C 0.9 -D 0.05 -U 1.0 -M 0.1 -S -1.0 -c -1","-R 2-3");
//       model.mineAssociationRules();
//       System.out.println(model);
//       MyFPGrowthModel model =new MyFPGrowthModel(
//                                        "C:\\Users\\Admin\\Desktop\\data\\weather.nominal.arff",
//                                        "-P 2 -I -1 -N 10 -T 0 -C 0.9 -D 0.05 -U 1.0 -M 0.1",
//                                        "-N -R first-last");
//      model.mineAssociationRule();
//       System.out.println(model);
//
//       MyKnowledgeMo del model=new MyKnowledgeModel(
//               "D:\\weka\\Weka-3-8-5\\data\\iris.arff",null,null);
//       model.trainset=model.divideTrainTestR(model.dataset, 70, false);
//       model.testset=model.divideTrainTestR(model.dataset, 70, true);
//       System.out.println(model);
//       System.out.println(model.trainset.toSummaryString());
//       System.out.println(model.testset.toSummaryString());
////bai cay
//  MyDecisionTreeModel model = new MyDecisionTreeModel(
//          "C:\\Users\\Admin\\Desktop\\data\\iris.arff","-C 0.25 -M 2",null);
//          model.buildDecisionTree();
//          model.evaluateDecisionTree();
//          System.out.println(model);
////          model.saveModel("C:\\Users\\Admin\\Desktop\\data\\decisiontree.model", model.tree);
//           model.tree = (J48)model.loadModel("C:\\Users\\Admin\\Desktop\\data\\decisiontree.model");
//           model.predictClassLabel(model.testset);
//bai 8
//        MyNaiveBayesModel model = new MyNaiveBayesModel();
//        model.builNaiveBayes("C:\\Users\\Admin\\Desktop\\data-exp\\iris_train.arff");
//        model.evaluateNaivebayes("C:\\Users\\Admin\\Desktop\\data-exp\\iris_test.arff");
//        model.predictClassLabel("C:\\Users\\Admin\\Desktop\\data-exp\\iris_unlabel.arff", "C:\\Users\\Admin\\Desktop\\data-exp\\iris_predict_nb.arff");
//        System.out.println(model);
 ///bai 9
//    MyNeuralNetwordModel model = new MyNeuralNetwordModel("","-L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a",null);
//        model.buildNeuralNetwork("C:\\Users\\Admin\\Desktop\\data-exp\\iris_train.arff");
//        model.evaluateNeuralNetwork("C:\\Users\\Admin\\Desktop\\data-exp\\iris_test.arff");
//        model.predictClassLabel("C:\\Users\\Admin\\Desktop\\data-exp\\iris_unlabel.arff", "C:\\Users\\Admin\\Desktop\\data-exp\\iris_predict_ann.arff");
//        System.out.println(model);
//    

//bai10
//   MySVMModel model = new MySVMModel ("",
//           "-C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007\" -calibrator \"weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4\"",
//           null);
//   model.buildSVM("C:\\Users\\Admin\\Desktop\\data-exp\\iris_train.arff");
//   model.evaluateSVM("C:\\Users\\Admin\\Desktop\\data-exp\\iris_test.arff");
//   model.predictClassLabel("C:\\Users\\Admin\\Desktop\\data-exp\\iris_unlabel.arff", 
//           "C:\\Users\\Admin\\Desktop\\data-exp\\iris_predict_svm.arff");
//   System.out.println(model);


////bai 11
//  MyKNNModel model = new MyKNNModel ("",
//           "-K 3 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -R first-last\\\"\"",
//           null);
//   model.buildkNN("C:\\Users\\Admin\\Desktop\\data-exp\\iris_train.arff");
//   model.evaluatekNN("C:\\Users\\Admin\\Desktop\\data-exp\\iris_test.arff");
//   model.predictClassLabel("C:\\Users\\Admin\\Desktop\\data-exp\\iris_unlabel.arff", 
//           "C:\\Users\\Admin\\Desktop\\data-exp\\iris_predict_knn.arff");
//   System.out.println(model);
   
//  //phuong pháp Bagging
//     MyBaggingModel model = new MyBaggingModel("",null,null);
//     model.buildMyBaggingModel("C:\\Users\\Admin\\Desktop\\data-exp\\iris_train.arff");
//     model.evaluateBaggingModel("C:\\Users\\Admin\\Desktop\\data-exp\\iris_test.arff");
//     model.predictClassLabel("C:\\Users\\Admin\\Desktop\\data-exp\\iris_unlabel.arff", 
//               "C:\\Users\\Admin\\Desktop\\data-exp\\iris_predict_bag.arff");
//     System.out.println("Finished");

////phuong pháp boosting
//        MyBoostingModel model = new MyBoostingModel("",null,null);
//        model.buildMyBoostingModel("C:\\Users\\Admin\\Desktop\\data-exp\\iris_train.arff");
//        model.evaluateBoostingModel("C:\\Users\\Admin\\Desktop\\data-exp\\iris_test.arff");
//        model.predictClassLabel("C:\\Users\\Admin\\Desktop\\data-exp\\iris_unlabel.arff", 
//               "C:\\Users\\Admin\\Desktop\\data-exp\\iris_predict_boost.arff");
//        System.out.println("Finished");


//     //phuong phap voting
//        MyVotingModel model = new MyVotingModel("",null,null);
//        model.buildMyVotingModel("C:\\Users\\Admin\\Desktop\\data-exp\\iris_train.arff");
//        model.evaluateVotingModel("C:\\Users\\Admin\\Desktop\\data-exp\\iris_test.arff");
//        model.predictClassLabel("C:\\Users\\Admin\\Desktop\\data-exp\\iris_unlabel.arff", 
//               "C:\\Users\\Admin\\Desktop\\data-exp\\iris_predict_vote .arff");
//        System.out.println("Finished");


        //phuongphap stacking
        
        MyBlendingModel model = new MyBlendingModel("",null,null);
        model.buildMyBlendingModel("C:\\Users\\Admin\\Desktop\\data-exp\\iris_train.arff");
        model.evaluateBlendingModel("C:\\Users\\Admin\\Desktop\\data-exp\\iris_test.arff");
        model.predictClassLabel("C:\\Users\\Admin\\Desktop\\data-exp\\iris_unlabel.arff", 
               "C:\\Users\\Admin\\Desktop\\data-exp\\iris_predict_blending.arff");
        System.out.println("Finished");
    }

}
