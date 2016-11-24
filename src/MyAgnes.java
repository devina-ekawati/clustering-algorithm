import weka.clusterers.AbstractClusterer;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;

/**
 * Created by Tifani on 11/19/2016.
 */
public class MyAgnes extends AbstractClusterer {
    /** link types */
    public static final int SINGLE = 0;
    public static final int COMPLETE = 1;

    private int numCluster = 2;
    private int linkType = SINGLE;
    private Instances instances;

    private int nCurrentCluster;
    private EuclideanDistance euclideanDistance;
    private ArrayList<ArrayList<Integer>> clusters;
    private ArrayList<Node> hierarchy;

    public MyAgnes(int numCluster, int linkType) {
        this.numCluster = Math.max(1, numCluster);
        this.linkType = linkType;
    }

    @Override
    public void buildClusterer(Instances instances) throws Exception {
        this.instances = instances;
        nCurrentCluster = instances.numInstances();
        if (nCurrentCluster == 0) return; //no data

        this.euclideanDistance = new EuclideanDistance(instances);
        clusters = new ArrayList<>(nCurrentCluster);
        hierarchy = new ArrayList<>(nCurrentCluster);
        for(int i=0; i<nCurrentCluster; i++) {
            ArrayList<Integer> currentCluster = new ArrayList<>();
            currentCluster.add(i);
            clusters.add(currentCluster);
            Node node = new Node();
            node.leftInstance = i;
            hierarchy.add(node);
        }
        combineCluster();
    }

    @Override
    public int numberOfClusters() throws Exception {
        return numCluster;
    }

    private void combineCluster() throws Exception {
        int clusterA = 0;
        int clusterB = 1;
        double minDistance = calculateClusterDistance(clusters.get(clusterA), clusters.get(clusterB));

        for(int i=0; i<clusters.size()-1; i++) {
            for (int j=i+1; j<clusters.size(); j++) {
                double newDistance = calculateClusterDistance(clusters.get(i), clusters.get(j));
                if (newDistance <= minDistance) {
                    minDistance = newDistance;
                    clusterA = i;
                    clusterB = j;
                }
            }
        }

        // combine cluster
        Node nodeA = hierarchy.get(clusterA);
        Node nodeB = hierarchy.get(clusterB);
        ArrayList<Integer> clusterMemberA = clusters.get(clusterA);
        ArrayList<Integer> clusterMemberB = clusters.get(clusterB);

        // node
        Node parent = new Node(nodeA, nodeB, minDistance);
        hierarchy.set(clusterA, parent);
        hierarchy.remove(clusterB);

        // cluster
        for(Integer instanceID: clusterMemberB) {
            clusterMemberA.add(instanceID);
        }
        clusters.remove(clusterMemberB);
        nCurrentCluster--;

        if (nCurrentCluster > numberOfClusters()) {
            combineCluster();
        }
    }

    private double calculateClusterDistance(ArrayList<Integer> clusterA, ArrayList<Integer> clusterB) {
        double distance = euclideanDistance.distance(
                instances.instance(clusterA.get(0)),
                instances.instance(clusterB.get(0)));
        for(int i=0; i<clusterA.size(); i++) {
            for(int j=0; j<clusterB.size(); j++) {
                double newDistance = euclideanDistance.distance(
                        instances.instance(clusterA.get(i)),
                        instances.instance(clusterB.get(j)));
                if (linkType == SINGLE && newDistance < distance) {
                    distance = newDistance;
                } else if (linkType == COMPLETE && newDistance > distance) {
                    distance = newDistance;
                }
            }
        }
        return  distance;
    }

    @Override
    public int clusterInstance(Instance instance) throws Exception {
        if (clusters.size() == 0) {
            return 0;
        } else {
            ArrayList<Integer> newCluster = new ArrayList<>();
            newCluster.add(instances.size());
            instances.add(instances.size(), instance);
            double distance = calculateClusterDistance(newCluster, clusters.get(0));
            int cluster = 0;
            for(int i=0; i<clusters.size(); i++) {
                double newDistance = calculateClusterDistance(newCluster, clusters.get(i));
                if (newDistance < distance) {
                    distance = newDistance;
                    cluster = i;
                }
            }
            instances.delete(instances.size()-1);
            return cluster;
        }
    }

    @Override
    public String toString() {
        int attIdx = instances.classIndex();
        StringBuffer stringBuffer = new StringBuffer();
        try {
            if (numberOfClusters() > 0) {
                for(int i=0; i<hierarchy.size(); i++) {
                    if (hierarchy.get(i) != null) {
                        String result = hierarchy.get(i).toString(attIdx);
                        if (result != null) {
                            stringBuffer.append("Cluster " + i + ": [");
                            stringBuffer.append(clusters.get(i).get(0));
                            for(int j=1; j<clusters.get(i).size(); j++) {
                                stringBuffer.append(", ");
                                stringBuffer.append(clusters.get(i).get(j));
                            }
                            stringBuffer.append("]\n");
                            stringBuffer.append(result);
                            stringBuffer.append("\n\n");
                        }
                    }
                }
                stringBuffer.append("\n");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return stringBuffer.toString();
    }

    class Node {
        Node left = null;
        Node right = null;
        Node parent = null;
        int leftInstance;
        int rightInstance;
        double leftDistance = 0;
        double rightDistance = 0;
        double position = 0;

        public Node() {

        }

        public Node(Node left, Node right, double distance) {
            if (left.isLeaf()) {
                this.leftInstance = left.leftInstance;
                this.leftDistance = distance;
            } else {
                this.left = left;
                this.left.parent = this;
                this.leftDistance = distance - left.position;
            }

            if (right.isLeaf()) {
                this.rightInstance = right.leftInstance;
                this.rightDistance = distance;
            } else {
                this.right = right;
                this.right.parent = this;
                this.rightDistance = distance - right.position;
            }

            this.position = distance;
        }

        private boolean isLeaf() {
            return (position == 0);
        }

        public String toString(int attIndex) {
            NumberFormat nf = NumberFormat.getNumberInstance(new Locale("en", "US"));
            DecimalFormat decimalFormat = (DecimalFormat) nf;
            decimalFormat.applyPattern("#.#####");

            if (parent == null && left == null && right ==null) {
                return null;
            }
            
            if (left == null) {
                if (right == null) { // left == null, right == null
                    return "("
                            + instances.instance(leftInstance).value(attIndex) + ":"
                            + decimalFormat.format(leftDistance) + ","
                            + instances.instance(rightInstance).value(attIndex)
                            + ":" + decimalFormat.format(rightDistance) + ")";
                } else { // left == null, right != null
                    return "("
                            + instances.instance(leftInstance).value(attIndex) + ":"
                            + decimalFormat.format(leftDistance) + ","
                            + right.toString(attIndex) + ":"
                            + decimalFormat.format(rightDistance) + ")";
                }
            } else {
                if (right == null) { // left != null, right == null
                    return "(" + left.toString(attIndex) + ":"
                            + decimalFormat.format(leftDistance) + ","
                            + instances.instance(rightInstance).value(attIndex)
                            + ":" + decimalFormat.format(rightDistance) + ")";
                } else { // left != null, right != null
                    return "(" + left.toString(attIndex) + ":"
                            + decimalFormat.format(leftDistance) + ","
                            + right.toString(attIndex) + ":"
                            + decimalFormat.format(rightDistance) + ")";
                }
            }
        }
    }
}
