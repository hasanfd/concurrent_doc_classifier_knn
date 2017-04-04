package com.machinelearning.doc_classifier.knn.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.machinelearning.doc_classifier.document.Document;
import com.machinelearning.doc_classifier.document.DocumentCache;
import com.machinelearning.doc_classifier.knn.Utils;
import com.machinelearning.doc_classifier.knn.category_classifier.KNNCategoryClassifier;
import com.machinelearning.doc_classifier.knn.similarity.DocumentSimilarityMethod;

public class KNNClassifyTestDocumentThread extends Thread {

	private static final Logger logger = Logger.getLogger( KNNClassifyTestDocumentThread.class.getName() );
	
	private String testDocumentName;
	private DocumentCache documentCache;
	private DocumentSimilarityMethod documentSimilarityMethod;
	private KNNCategoryClassifier categoryClassifier;
	private String[] categoryNames;
	private int k;
	
	public KNNClassifyTestDocumentThread(String testDocumentName, 
			DocumentCache documentCache, 
			DocumentSimilarityMethod documentSimilarityMethod, 
			KNNCategoryClassifier categoryClassifier, 
			String[] categoryNames, 
			int k) {
		this.testDocumentName = testDocumentName;
		this.documentCache = documentCache;
		this.documentSimilarityMethod = documentSimilarityMethod;
		this.categoryClassifier = categoryClassifier;
		this.categoryNames = categoryNames;
		this.k = k;
	}
	
	public void run() {
		Document testDocument = documentCache.getTestDocument(testDocumentName);		
		List<KNNClassificationInfo> results = new ArrayList<KNNClassificationInfo>();
		
		Set<String> trainDocumentNames = documentCache.getTrainDocumentNames();
		for(String trainDocumentName : trainDocumentNames) {
			Document trainDocument = documentCache.getTrainDocument(trainDocumentName);
			double similarity = documentSimilarityMethod.calculateSimilarity(testDocument,trainDocument);
			
			KNNClassificationInfo classifierResult = new KNNClassificationInfo(trainDocumentName,
					Utils.getCategoryFromFileName(testDocumentName),
					Utils.getCategoryFromFileName(trainDocumentName),
					similarity);

			results.add(classifierResult);
		}
		
		String estimatedCategory = categoryClassifier.estimate(results,categoryNames,k);
		logger.log(Level.INFO, "Document \t {0} \t classified as \t {1}", new Object[]{testDocumentName,estimatedCategory});
	}
	
}
