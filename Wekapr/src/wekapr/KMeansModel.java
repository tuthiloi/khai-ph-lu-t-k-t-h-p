package wekapr;

import weka.classifiers.Evaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import wekapr.MyKnowledgeModel;

/**
 *
 * @author Lợii
 */
public class KMeansModel extends MyKnowledgeModel {

    SimpleKMeans kmeans;
    Evaluation eval;

    //Cac constructor
    public KMeansModel() {
    }
    
    public KMeansModel(String filename, String m_opts, String d_opts) throws Exception {
        super(filename, m_opts, d_opts);
    }

    //Cac phuong thuc
    public void buildKMeansModel(String filename) throws Exception {
        //Doc train set vao bo nho
        setTrainset(filename);

        //Thiet lap mo hinh kmeans
        kmeans = new SimpleKMeans();
        kmeans.setNumClusters(3);
        kmeans.setDistanceFunction(new EuclideanDistance());
        kmeans.buildClusterer(trainset);
        //Xuat thong so cua mo hinh ra man hinh
        System.out.println(kmeans);
    }
    
    public void predictCluster(String filename) throws Exception {
        DataSource ds = new DataSource(filename);
        Instances unlabel = ds.getDataSet();
        unlabel.setClassIndex(unlabel.numAttributes() - 1);
        //predict label for each instance
        for (int i = 0; i < unlabel.numInstances(); i++) {
            double predict = kmeans.clusterInstance(unlabel.instance(i));
            System.out.println("Instance " + i + " belongs to cluster " + predict);
        }
    }
}
