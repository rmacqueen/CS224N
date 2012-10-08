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
public class IBMModel1 implements WordAligner {

  private static final long serialVersionUID = 1315751943476440515L;
  
  // TODO: Use arrays or Counters for collecting sufficient statistics
  // from the training data.
  private CounterMap<String,String> sourceTargetProbs;
  private Counter<String> sourceCounts;
  private Counter<String> targetCounts;
  private double numSentences = 0;

  public Alignment align(SentencePair sentencePair) {

    Alignment alignment = new Alignment();

    for(int english_index = 0; english_index < sentencePair.getTargetWords().size(); english_index++){
      String english_word = sentencePair.getTargetWords().get(english_index);
      double french_max = sourceTargetProbs.getCount(english_word, NULL_WORD);
      int french_match = -1;
      for(int f_index = 0; f_index < sentencePair.getSourceWords().size(); f_index++){
        String french_word = sentencePair.getSourceWords().get(f_index);

        double nextProb = sourceTargetProbs.getCount(english_word, french_word);
        if(nextProb > french_max){
          french_max = nextProb;
          french_match = f_index;
        }
      }
      if(french_match >= 0)alignment.addPredictedAlignment(english_index, french_match);         
    }
    
    return alignment;
  }

  public void train(List<SentencePair> trainingPairs) {

    System.out.println("IBMModel1 beginning training....");

    CounterMap<String, String> t = new CounterMap<String,String>();
    targetCounts = new Counter<String>();
    sourceCounts = new Counter<String>();
    numSentences = trainingPairs.size();
    Set<String> foreign = new HashSet<String>();
    Set<String> english = new HashSet<String>();
    for(SentencePair pair : trainingPairs){
      List<String> sourceWords = pair.getSourceWords();
      List<String> targetWords = pair.getTargetWords();
      for(int i = 0; i < sourceWords.size(); i++){
          //System.out.println(sourceWords.get(i));
         foreign.add(sourceWords.get(i));
      }
      for(int j = 0; j < targetWords.size(); j++){
         english.add(targetWords.get(j));
      }
    }

    foreign.add(NULL_WORD);
    double num_probs = english.size() * foreign.size();


    for(String f : foreign){
      for(String e : english){
        t.setCount(e, f, 1.0 / english.size());
      }
    }
    Counter<String> s_total = new Counter<String>();
    boolean converge = false;
    int iteration_count = 0;
    while(!converge){
       CounterMap<String, String> count = new CounterMap<String, String>();
       Counter<String> total = new Counter<String>();

       for(SentencePair pair : trainingPairs){
          pair.getSourceWords().add(0, NULL_WORD);
          for(String english_word : pair.getTargetWords()){
            s_total.setCount(english_word , 0.0);
            for(String french_word : pair.getSourceWords()){
              s_total.incrementCount(english_word, t.getCount(english_word, french_word));
            }
          }

          for(String english_word : pair.getTargetWords()){
            for(String french_word : pair.getSourceWords()){
              double cnt = t.getCount(english_word, french_word) / s_total.getCount(english_word);
              count.incrementCount(english_word, french_word, cnt);
              total.incrementCount(french_word, cnt);
            }
          }
          pair.getSourceWords().remove(0);
       }


      int num_converged = 0;
      for (String french_word : foreign){
        for(String english_word : english){
            double new_prob = count.getCount(english_word, french_word) / total.getCount(french_word);
            double delta = Math.abs(t.getCount(english_word, french_word) - new_prob);
            
            if(delta < 0.01){
              num_converged += 1; 
            }else{
              //System.out.println("delta: " + delta + " - " + english_word + " : " + french_word);
            }
            t.setCount(english_word, french_word, new_prob);
        }
      }
      iteration_count++;
      if (num_converged == num_probs) converge = true;
      System.out.print("After " + iteration_count+ " runs, "+ num_converged + "/" + num_probs + " matches have converged."); 
      System.out.println(" We are " + (100 * ((double)num_converged / (double)num_probs)) + " % of the way there!");
    }
    sourceTargetProbs = t;
  }
}
