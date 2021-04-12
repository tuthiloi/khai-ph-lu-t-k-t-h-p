package wekapr;

import weka.clusterers.EM;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import wekapr.MyKnowledgeModel;
import weka.classifiers.Classifier;

/**
 *
 * @author asus
 */
public class MyEMModel extends MyKnowledgeModel {
    EM em;
    
    //Cac constructors

    public MyEMModel() {
    }

    public MyEMModel(String filename, String m_opts, String d_opts) throws Exception {
        super(filename, m_opts, d_opts);
    }
    
    //cac phuong thuc
    public void buildEMModel(String filename) throws Exception{
        //Doc train set vao bo nho
        setTrainset(filename);
        
        //thiet lap mo hinh EM
        em = new EM();
        em.buildClusterer(trainset);
    }
    
    public void predictCluster(String filename) throws Exception{
        //doc du lieu vao bo nho
        DataSource ds = new DataSource(filename);
        Instances unlabel = ds.getDataSet();
        //Du doan cluster
        for (int i = 0; i < unlabel.numInstances(); i++) {
            double predict = em.clusterInstance(unlabel.instance(i));
            System.out.println("Instance " + i + " belongs to cluster " + predict);
        }
    }
}
