package com.cyfirma.core.service.impl;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.mongodb.BasicDBObject;
import com.mongodb.DB;
import com.mongodb.DBCollection;
import com.mongodb.DBCursor;
import com.mongodb.MongoClient;

import opennlp.tools.cmdline.parser.ParserTool;
import opennlp.tools.parser.Parse;
import opennlp.tools.parser.Parser;
import opennlp.tools.parser.ParserFactory;
import opennlp.tools.parser.ParserModel;

//extract noun phrases from a single sentence using OpenNLP
public class ExtractNounPhraseNlp {

	static String sentence = "Ram is a boy Ram Ram";

	static Set<String> nounPhrases = new HashSet<>();
	static List<String> wordList = new ArrayList<String>();
	static String wordliststring = "";
	static Map<String, Integer> wordCounter = new LinkedHashMap<String, Integer>();
	
	static List descpript;
	static List dbDataList;

	public static void main(String[] args) {

		dbDataList = getDbConnection();

		InputStream modelInParse = null;
		for (int i = 0; i < descpript.size(); ++i) {

			System.out.println("List Value" + descpript.get(i));
		}
		try {
			// load chunking model
			modelInParse = new FileInputStream("en-parser-chunking.bin");
			ParserModel model = new ParserModel(modelInParse);

			// create parse tree
			Parser parser = ParserFactory.create(model);
			Parse topParses[] = ParserTool.parseLine(descpript.toString().toLowerCase().replaceAll("[^a-zA-Z0-9]", " "), parser, 1);
			Parse words[] = null;
			// call subroutine to extract noun phrases
			for (Parse nodes : topParses) {
				words = nodes.getTagNodes(); // we will get a list of nodes

			}

			for (Parse word : words) {
				// Change the types according to your desired types
				if (word.getType().equals("NN") || word.getType().equals("NNP") || word.getType().equals("NNS")) {
					// System.out.println(word.toString());
					wordList.add(word.toString());
					// wordliststring(word.toString());

				}
			}
			// System.out.println("Word lIst Value "+ wordList);

			for (String text : wordList) {
				// System.out.println(text);
				countOccurences(wordList, text);
			}
			// Create a list from elements of HashMap 
	        List<Map.Entry<String, Integer> > list = 
	               new LinkedList<Map.Entry<String, Integer> >(wordCounter.entrySet()); 
	  
	        // Sort the list 
	        Collections.sort(list, new Comparator<Map.Entry<String, Integer> >() { 
	            public int compare(Map.Entry<String, Integer> o1,  
	                               Map.Entry<String, Integer> o2) 
	            { 
	                return (o1.getValue()).compareTo(o2.getValue()); 
	            } 
	        }); 
			for (Map.Entry<String, Integer> entry : list) {
				System.out.println(entry.getKey() + ":" + entry.getValue());
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (modelInParse != null) {
				try {
					modelInParse.close();
				} catch (IOException e) {
				}
			}
		}
	}

	public static List<String> getDbConnection() {

		descpript = new ArrayList<String>();
		try {
			MongoClient mongo = new MongoClient("localhost", 27017);
			DB db = mongo.getDB("core");
			DBCollection col = db.getCollection("rss_feed_entry");
			System.out.println("\n1. Get 'name' field only");
			BasicDBObject allQuery = new BasicDBObject();
			BasicDBObject fields = new BasicDBObject();
			BasicDBObject neQuery = new BasicDBObject();
			// neQuery.put("description", new BasicDBObject("$lt", 5));
			fields.put("_id", 0);
			fields.put("description", 1);
			// DBCursor cursor2 = col.find(neQuery);
			DBCursor cursor2 = col.find(allQuery, fields).limit(3);

			while (cursor2.hasNext()) {
				// System.out.println(cursor2.next());
				descpript.add(cursor2.next());
			}
//		for(int i = 0; i< descpript.size() ; ++i) {
//		
//		System.out.println("List Value"+descpript.get(i));
//		}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return dbDataList;

	}

	static int countOccurences(List<String> words, String word) {
		// search for pattern in a
		int count = 0;

		for (String text : words) {
			// if match found increase count
			if (text.equalsIgnoreCase(word)) {
				count++;
				wordCounter.put(word, count);
				// System.out.println(word+" count is"+count);
			}
			// System.out.println(words);
		}

		return count;
	}
}
