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
public class IBMModel2 implements WordAligner {

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

  private String getKey(String e_word, String f_word, int l, int m){
    return (e_word + f_word + Integer.toString(l) + Integer.toString(m));

  }

  private void debug(String line){
    System.out.println(line);
  }

  private void debug(double line){
    System.out.println(line);
  }

  private void debug(int line){
    System.out.println(line);
  }

  public void train(List<SentencePair> trainingPairs) {

    debug("IBMModel 2 beginning training....");

    CounterMap<String, String> t = new CounterMap<String,String>();

    Counter<String> q = new Counter<String>();

    targetCounts = new Counter<String>();
    sourceCounts = new Counter<String>();
    numSentences = trainingPairs.size();
    Set<String> foreign = new HashSet<String>();
    Set<String> english = new HashSet<String>();


    int max_m = 0;
    int max_l = 0;


    for(SentencePair pair : trainingPairs){
      List<String> sourceWords = pair.getSourceWords();
      List<String> targetWords = pair.getTargetWords();

      max_m = Math.max(sourceWords.size(), max_m);
      max_l = Math.max(targetWords.size(), max_l);

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



    for(SentencePair pair : trainingPairs){
      List<String> sourceWords = pair.getSourceWords();
      List<String> targetWords = pair.getTargetWords();
      for(String e : targetWords){
        for(String f : sourceWords){
          
          t.setCount(e, f, Math.random());
          //debug(max_m);
          //debug(max_l);
          for(int i = 0; i < max_m; i++){
            for(int j = 0; j < max_l; j++){
              debug(e);
              //debug(getKey(e, f, j, i));
              q.setCount(getKey(e, f, j, i), Math.random());
            }
          }
        }
      }
    }

    Counter<String> s_total = new Counter<String>();
    boolean converge = false;
    int iteration_count = 0;

    debug("Starting iterations.......");

    while(!converge){
       CounterMap<String, String> count = new CounterMap<String, String>();
       Counter<String> total = new Counter<String>();

       Counter<String> count_lens = new Counter<String>();
       Counter<String> total_lens = new Counter<String>();

       

       for(SentencePair pair : trainingPairs){

          int l = pair.getTargetWords().size();
          int m = pair.getSourceWords().size();

          pair.getSourceWords().add(0, NULL_WORD);
          for(String english_word : pair.getTargetWords()){
            s_total.setCount(english_word , 0.0);
            for(String french_word : pair.getSourceWords()){
              String key_count_lens = getKey(english_word, french_word, l, m);
              s_total.incrementCount(english_word, t.getCount(english_word, french_word) * q.getCount(key_count_lens));
            }
          }

          for(String english_word : pair.getTargetWords()){
            for(String french_word : pair.getSourceWords()){
              String key_count_lens = getKey(english_word, french_word, l, m);
              double cnt = (t.getCount(english_word, french_word) * q.getCount(key_count_lens)) / s_total.getCount(english_word);
              
              //debug(s_total.getCount(english_word));

              count.incrementCount(english_word, french_word, cnt);
              total.incrementCount(french_word, cnt);

              count_lens.incrementCount(key_count_lens, cnt);

              String key_total_lens = french_word + Integer.toString(pair.getTargetWords().size()) + Integer.toString(pair.getSourceWords().size());

              total_lens.incrementCount(key_total_lens, cnt);
            }
          }
          pair.getSourceWords().remove(0);
       }


       for(SentencePair pair : trainingPairs){

          int l = pair.getTargetWords().size();
          int m = pair.getSourceWords().size();

          for(String english_word : pair.getTargetWords()){
            for(String french_word : pair.getSourceWords()){
              String key_count_lens = getKey(english_word, french_word, l, m);
              String key_total_lens = french_word + Integer.toString(pair.getTargetWords().size()) + Integer.toString(pair.getSourceWords().size());
              q.setCount(key_count_lens, count_lens.getCount(key_count_lens) / total_lens.getCount(key_total_lens));
            }
          }
       }


      int num_converged = 0;
      for (String french_word : foreign){
        for(String english_word : english){
            //debug(count.getCount(english_word, french_word));
            double new_prob = count.getCount(english_word, french_word) / total.getCount(french_word);
            double delta = Math.abs(t.getCount(english_word, french_word) - new_prob);
            
            if(delta < 0.01){
              num_converged += 1; 
            }else{
              //System.out.println("newprob " + new_prob);
            }
            t.setCount(english_word, french_word, new_prob);
        }
      }
      iteration_count++;
      if (num_converged == num_probs) converge = true;
      System.out.print("After " + iteration_count+ " runs, "+ num_converged + "/" + num_probs + " matches have converged."); 
      debug(" We are " + (100 * ((double)num_converged / (double)num_probs)) + " % of the way there!");
    }
    sourceTargetProbs = t;
  }
}
