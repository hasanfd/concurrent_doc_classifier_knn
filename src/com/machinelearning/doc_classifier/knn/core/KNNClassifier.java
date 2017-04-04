package com.machinelearning.doc_classifier.knn.core;

import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.machinelearning.doc_classifier.document.DocumentCache;
import com.machinelearning.doc_classifier.knn.category_classifier.KNNCategoryClassifier;
import com.machinelearning.doc_classifier.knn.category_classifier.MajorityVoteClassifier;
import com.machinelearning.doc_classifier.knn.similarity.CosineSimilarity;
import com.machinelearning.doc_classifier.knn.similarity.DocumentSimilarityMethod;

public class KNNClassifier {

	private static final Logger logger = Logger.getLogger( KNNClassifier.class.getName() );
	
	private String categoryNames[];
	private int k;
	private DocumentCache documentCache;
	
	private KNNCategoryClassifier categoryClassifier = new MajorityVoteClassifier();
	private DocumentSimilarityMethod documentSimilarityMethod = new CosineSimilarity();
	
	public KNNClassifier(String[] categoryNames, int k, DocumentCache documentCache) {
		this.categoryNames = categoryNames;
		this.k = k;
		this.documentCache = documentCache;
	}
	
	public KNNClassifier(String[] categoryNames, int k, DocumentCache documentCache, KNNCategoryClassifier categoryClassifier,DocumentSimilarityMethod documentSimilarityMethod) {
		this.categoryNames = categoryNames;
		this.k = k;
		this.documentCache = documentCache;
		this.categoryClassifier = categoryClassifier;
		this.documentSimilarityMethod = documentSimilarityMethod;
	}

	public void classify() throws InterruptedException {
		
		long startTime = System.currentTimeMillis();
		
		ExecutorService es = Executors.newCachedThreadPool();
				
		Set<String> testDocuments = documentCache.getTestDocumentNames();
		for(String testDocumentName : testDocuments) {
			Thread thread = new KNNClassifyTestDocumentThread(testDocumentName, documentCache, documentSimilarityMethod,categoryClassifier, categoryNames,k);
			es.execute(thread);
		}
		
		es.shutdown();
		boolean finished = es.awaitTermination(100, TimeUnit.MINUTES);
		if(!finished) {
			logger.log(Level.SEVERE,"some threads not finished in maximum time!"); 
			return;
		}
		
		long stopTime = System.currentTimeMillis();
	    long elapsedTime = stopTime - startTime;
	    logger.log(Level.INFO, "all documents classified in {0} ms", new Object[]{elapsedTime});
	}
}