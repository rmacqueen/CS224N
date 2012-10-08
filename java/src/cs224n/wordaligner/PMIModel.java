package cs224n.wordaligner;  

import cs224n.util.*;
//import java.util.List;
import java.util.*;

/**
 * 
 *
 * 
 * IMPORTANT: Make sure that you read the comments in the
 * cs224n.wordaligner.WordAligner interface.
 * 
 */
public class PMIModel implements WordAligner {

  private static final long serialVersionUID = 1315751943476440515L;
  
  // TODO: Use arrays or Counters for collecting sufficient statistics
  // from the training data.
  private CounterMap<String,String> sourceTargetCounts;
  private Counter<String> sourceCounts;
  private Counter<String> targetCounts;
  private double numSentences = 0;

  public Alignment align(SentencePair sentencePair) {
    // Placeholder code below. 
    // TODO Implement an inference algorithm for Eq.1 in the assignment
    // handout to predict alignments based on the counts you collected with train().
     //System.out.print("------------Here in Align-------------------\n");
   //System.out.println(sentencePair.getSourceWords().get(0));
    Alignment alignment = new Alignment();
    int numSourceWords = sentencePair.getSourceWords().size();
    int numTargetWords = sentencePair.getTargetWords().size();
    for(int trgtIndex = 0; trgtIndex < numTargetWords; trgtIndex++){
	String tgtWord = sentencePair.getTargetWords().get(trgtIndex);
	double tgtProb = targetCounts.getCount(tgtWord) / targetCounts.totalCount();
	int bestIndex = -1;
	double bestMatch = -1.0;
     for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
	String srcWord = sentencePair.getSourceWords().get(srcIndex);
	double matchCount = sourceTargetCounts.getCount(srcWord, tgtWord);
	double probMatch = matchCount / numSentences;
	System.out.println(probMatch);
	double srcProb = sourceCounts.getCount(srcWord) / sourceCounts.totalCount();
	double a_i = probMatch / (srcProb * tgtProb);
	if(a_i > bestMatch){
	   bestMatch = a_i;
	   bestIndex = srcIndex;
	}
       /*int tgtIndex = srcIndex;
       if (tgtIndex < numTargetWords) {
        // Discard null alignments
         alignment.addPredictedAlignment(tgtIndex, srcIndex);
       }*/
     }
     //if(bestIndex==0)//System.out.print("NULL predicted\n");
	alignment.addPredictedAlignment(trgtIndex, bestIndex);
    }
    return alignment;
  }

  public void train(List<SentencePair> trainingPairs) {
    System.out.print("------------sdfsdfHere in training-------------------\n");
    sourceTargetCounts = new CounterMap<String,String>();
    targetCounts = new Counter<String>();
    sourceCounts = new Counter<String>();
    numSentences = trainingPairs.size();
    for(SentencePair pair : trainingPairs){
      List<String> targetWords = pair.getTargetWords();
      List<String> sourceWords = pair.getSourceWords();
      //sourceWords.add(0, "<NULL>"); //not sure about this NULL
      for(String s : sourceWords){
        sourceCounts.incrementCount(s, 1.0);
      }
      for(String t : targetWords){
        targetCounts.incrementCount(t, 1.0);
      }
      
      Set<String> s1 = new HashSet<String>();
      for(String source : sourceWords){
        if(s1.contains(source)){
          continue;
        }else{
          s1.add(source);
        }
        Set<String> s2 = new HashSet<String>();
        for(String target : targetWords){
          
          if(s2.contains(target)){
          continue;
          }else{
          s2.add(target);
          }
          // TODO: Warm-up. Your code here for collecting sufficient statistics.
          sourceTargetCounts.incrementCount(source, target, 1.0);
        }
      }
    }
  }
}
