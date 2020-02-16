package com.cyfirma.core.service.impl;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import opennlp.tools.cmdline.parser.ParserTool;
import opennlp.tools.parser.Parse;
import opennlp.tools.parser.Parser;
import opennlp.tools.parser.ParserFactory;
import opennlp.tools.parser.ParserModel;


//extract noun phrases from a single sentence using OpenNLP
	public class ExtractNounPhraseNlp {

		static String sentence = "Ram is a boy";
		
		static Set<String> nounPhrases = new HashSet<>();
		
		public static void main(String[] args) {

			InputStream modelInParse = null;
			try {
				//load chunking model
				modelInParse = new FileInputStream("en-parser-chunking.bin"); //from http://opennlp.sourceforge.net/models-1.5/
				ParserModel model = new ParserModel(modelInParse);
				
				//create parse tree
				Parser parser = ParserFactory.create(model);
				Parse topParses[] = ParserTool.parseLine(sentence, parser, 1);
				Parse words[] = null;
				//call subroutine to extract noun phrases
				for (Parse nodes : topParses) {
					words=nodes.getTagNodes(); // we will get a list of nodes

				}

for(Parse word:words){
//Change the types according to your desired types
    if(word.getType().equals("NN") || word.getType().equals("NNP") || word.getType().equals("NNS")){
            System.out.println(word);
            }
        }
					//	p.show();
				//	getNounPhrases(p);
				
				//print noun phrases
				for (String s : nounPhrases)
				    System.out.println(s);
				
				//The Call
				//the Wild?
				//The Call of the Wild? //punctuation remains on the end of sentence
				//the author of The Call of the Wild?
				//the author
			}
			catch (IOException e) {
			  e.printStackTrace();
			}
			finally {
			  if (modelInParse != null) {
			    try {
			    	modelInParse.close();
			    }
			    catch (IOException e) {
			    }
			  }
			}
		}
		
		//recursively loop through tree, extracting noun phrases
		public static void getNounPhrases(Parse p) {
				
		    if (p.getType().equals("NP")) { //NP=noun phrase
		         nounPhrases.add(p.getCoveredText());
		    }
		    for (Parse child : p.getChildren())
		         getNounPhrases(child);
		    
//		    public List<String> tag(String str) {
//		        final List<String> tagLemme = new ArrayList<String>();
//		        String[] tokens =tokenizer.tokenize(str);
//		          System.setProperty("treetagger.home", "parametresTreeTagger/TreeTagger");
//		        TreeTaggerWrapper tt = new TreeTaggerWrapper<String>();
//		        try {
//		            tt.setModel("parametresTreeTagger/english/english.par");
//		            tt.setHandler(new TokenHandler<String>(){
//		                    public void token(String token, String pos, String lemma) {
//		                            tagLemme.add(token + "_" + pos + "_" + lemma);
//		                            //System.out.println(token + "_" + pos + "_" + lemma);
//		                    }
//		            });
//		            tt.process(asList(tokens));
//		         } catch (IOException e) {
//		            e.printStackTrace();
//		          } catch (TreeTaggerException e) {
//		            e.printStackTrace();
//		        }
//		    finally {
//		            tt.destroy();
//		    }
//		        return tagLemme;
//		    }
		}
	}
