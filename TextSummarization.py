# Implementation from https://dev.to/davidisrawi/build-a-quick-summarizer-with-python-and-nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from bs4 import BeautifulSoup
import re
import pandas as pd
import os
#conda install -c anaconda beautifulsoup4
 



#text_str = open(r"C:/Users/tahsin.asif/OneDrive - Antuit India Private Limited/Asif/AI/TextSummarization/text.txt","r")
from pathlib import Path
text_str = Path('C:/Users/tahsin.asif/OneDrive - Antuit India Private Limited/Asif/AI/TextSummarization/text.txt').read_text(encoding='utf-8')

text_str1 = '''
Kaggle
search
Search
Competitions
Datasets
Notebooks
Discussion
Courses
mdtahsinasif

medal
  Dataset

Sentiment140 dataset with 1.6 million tweets
Sentiment analysis with tweets
Μαριος Μιχαηλιδης KazAnova
 •  updated 2 years ago (Version 2)
Data
Tasks
Kernels(57)
Discussion(4)
Activity
Metadata
Download (228 MB)
New Notebook
Usability8.8
License
Other (specified in description)
Tags
internet


, 
online communities


, 
linguistics


, 
social networks


, 
languages


Description
Context
This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .

Content
It contains the following 6 fields:

target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)

ids: The id of the tweet ( 2087)

date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)

flag: The query (lyx). If there is no query, then this value is NO_QUERY.

user: the user that tweeted (robotickilldozr)

text: the text of the tweet (Lyx is cool)

Acknowledgements
The official link regarding the dataset with resources about how it was generated is here The official paper detailing the approach is here

Citation: Go, A., Bhayani, R. and Huang, L., 2009. Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1(2009), p.12.

Inspiration
To detect severity from tweets. You may have a look at this.

Data (228 MB)
Data Sources
training.1600000.processed.noemoticon.csv
6 columns
About this file
This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 2 = neutral, 4 = positive) and they can be used to detect sentiment . It contains the following 6 fields:

target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
ids: The id of the tweet ( 2087)
date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
flag: The query (lyx). If there is no query, then this value is NO_QUERY.
user: the user that tweeted (robotickilldozr)
text: the text of the tweet (Lyx is cool)
The official link regarding the dataset with resources about how it was generated is here The official paper detailing the approach is here

According to the creators of the dataset:

"Our approach was unique because our training data was automatically created, as opposed to having humans manual annotate tweets. In our approach, we assume that any tweet with positive emoticons, like :), were positive, and tweets with negative emoticons, like :(, were negative. We used the Twitter Search API to collect these tweets by using keyword search"

citation: Go, A., Bhayani, R. and Huang, L., 2009. Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1(2009), p.12.

s

Columns
0target
1467810369id
Mon Apr 06 22:19:45 PDT 2009date
NO_QUERYflag
_TheSpecialOne_user
@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer. You shoulda got David Carr of Third Day to do it. ;Dtext
training.1600000.processed.noemoticon.csv (227.74 MB)
6 of 6 columns
Views

0
target
1467810369
id
Mon Apr 06 22:19:45 PDT 2009
date
NO_QUERY
flag
_TheSpecialOne_
user
@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer. You shoulda got David Carr of Third Day to do it. ;D
text
774362
unique values
NO_QUERY
100%
659775
unique values
1581465
unique values
1	0	1467810672	Mon Apr 06 22:19:49 PDT 2009	NO_QUERY	scotthamilton	is upset that he can't update his Facebook by texting it... and might cry as a result School today also. Blah!
2	0	1467810917	Mon Apr 06 22:19:53 PDT 2009	NO_QUERY	mattycus	@Kenichan I dived many times for the ball. Managed to save 50% The rest go out of bounds
3	0	1467811184	Mon Apr 06 22:19:57 PDT 2009	NO_QUERY	ElleCTF	my whole body feels itchy and like its on fire
4	0	1467811193	Mon Apr 06 22:19:57 PDT 2009	NO_QUERY	Karoli	@nationwideclass no, it's not behaving at all. i'm mad. why am i here? because I can't see you all over there.
5	0	1467811372	Mon Apr 06 22:20:00 PDT 2009	NO_QUERY	joy_wolf	@Kwesidei not the whole crew
6	0	1467811592	Mon Apr 06 22:20:03 PDT 2009	NO_QUERY	mybirch	Need a hug
7	0	1467811594	Mon Apr 06 22:20:03 PDT 2009	NO_QUERY	coZZ	@LOLTrish hey long time no see! Yes.. Rains a bit ,only a bit LOL , I'm fine thanks , how's you ?
8	0	1467811795	Mon Apr 06 22:20:05 PDT 2009	NO_QUERY	2Hood4Hollywood	@Tatiana_K nope they didn't have it
9	0	1467812025	Mon Apr 06 22:20:09 PDT 2009	NO_QUERY	mimismo	@twittera que me muera ?
10	0	1467812416	Mon Apr 06 22:20:16 PDT 2009	NO_QUERY	erinx3leannexo	spring break in plain city... it's snowing
11	0	1467812579	Mon Apr 06 22:20:17 PDT 2009	NO_QUERY	pardonlauren	I just re-pierced my ears
12	0	1467812723	Mon Apr 06 22:20:19 PDT 2009	NO_QUERY	TLeC	@caregiving I couldn't bear to watch it. And I thought the UA loss was embarrassing . . . . .
13	0	1467812771	Mon Apr 06 22:20:19 PDT 2009	NO_QUERY	robrobbierobert	@octolinz16 It it counts, idk why I did either. you never talk to me anymore
14	0	1467812784	Mon Apr 06 22:20:20 PDT 2009	NO_QUERY	bayofwolves	@smarrison i would've been the first, but i didn't have a gun. not really though, zac snyder's just a doucheclown.
15	0	1467812799	Mon Apr 06 22:20:20 PDT 2009	NO_QUERY	HairByJess	@iamjazzyfizzle I wish I got to watch it with you!! I miss you and @iamlilnicki how was the premiere?!
16	0	1467812964	Mon Apr 06 22:20:22 PDT 2009	NO_QUERY	lovesongwriter	Hollis' death scene will hurt me severely to watch on film wry is directors cut not out now?
17	0	1467813137	Mon Apr 06 22:20:25 PDT 2009	NO_QUERY	armotley	about to file taxes
18	0	1467813579	Mon Apr 06 22:20:31 PDT 2009	NO_QUERY	starkissed	@LettyA ahh ive always wanted to see rent love the soundtrack!!
19	0	1467813782	Mon Apr 06 22:20:34 PDT 2009	NO_QUERY	gi_gi_bee	@FakerPattyPattz Oh dear. Were you drinking out of the forgotten table drinks?
20	0	1467813985	Mon Apr 06 22:20:37 PDT 2009	NO_QUERY	quanvu	@alydesigns i was out most of the day so didn't get much done
21	0	1467813992	Mon Apr 06 22:20:38 PDT 2009	NO_QUERY	swinspeedx	one of my friend called me, and asked to meet with her at Mid Valley today...but i've no time *sigh*
22	0	1467814119	Mon Apr 06 22:20:40 PDT 2009	NO_QUERY	cooliodoc	@angry_barista I baked you a cake but I ated it
23	0	1467814180	Mon Apr 06 22:20:40 PDT 2009	NO_QUERY	viJILLante	this week is not going as i had hoped
24	0	1467814192	Mon Apr 06 22:20:41 PDT 2009	NO_QUERY	Ljelli3166	blagh class at 8 tomorrow
25	0	1467814438	Mon Apr 06 22:20:44 PDT 2009	NO_QUERY	ChicagoCubbie	I hate when I have to call and wake people up
26	0	1467814783	Mon Apr 06 22:20:50 PDT 2009	NO_QUERY	KatieAngell	Just going to cry myself to sleep after watching Marley and Me.
27	0	1467814883	Mon Apr 06 22:20:52 PDT 2009	NO_QUERY	gagoo	im sad now Miss.Lilly
28	0	1467815199	Mon Apr 06 22:20:56 PDT 2009	NO_QUERY	abel209	ooooh.... LOL that leslie.... and ok I won't do it again so leslie won't get mad again
29	0	1467815753	Mon Apr 06 22:21:04 PDT 2009	NO_QUERY	BaptisteTheFool	Meh... Almost Lover is the exception... this track gets me depressed every time.
30	0	1467815923	Mon Apr 06 22:21:07 PDT 2009	NO_QUERY	fatkat309	some1 hacked my account on aim now i have to make a new one
31	0	1467815924	Mon Apr 06 22:21:07 PDT 2009	NO_QUERY	EmCDL	@alielayus I want to go to promote GEAR AND GROOVE but unfornately no ride there I may b going to the one in Anaheim in May though
32	0	1467815988	Mon Apr 06 22:21:09 PDT 2009	NO_QUERY	merisssa	thought sleeping in was an option tomorrow but realizing that it now is not. evaluations in the morning and work in the afternoon!
33	0	1467816149	Mon Apr 06 22:21:11 PDT 2009	NO_QUERY	Pbearfox	@julieebaby awe i love you too!!!! 1 am here i miss you
34	0	1467816665	Mon Apr 06 22:21:21 PDT 2009	NO_QUERY	jsoo	@HumpNinja I cry my asian eyes to sleep at night
35	0	1467816749	Mon Apr 06 22:21:20 PDT 2009	NO_QUERY	scarletletterm	ok I'm sick and spent an hour sitting in the shower cause I was too sick to stand and held back the puke like a champ. BED now
36	0	1467817225	Mon Apr 06 22:21:27 PDT 2009	NO_QUERY	crosland_12	@cocomix04 ill tell ya the story later not a good day and ill be workin for like three more hours...
37	0	1467817374	Mon Apr 06 22:21:30 PDT 2009	NO_QUERY	ajaxpro	@MissXu sorry! bed time came here (GMT+1) http://is.gd/fNge
38	0	1467817502	Mon Apr 06 22:21:32 PDT 2009	NO_QUERY	Tmttq86	@fleurylis I don't either. Its depressing. I don't think I even want to know about the kids in suitcases.
39	0	1467818007	Mon Apr 06 22:21:39 PDT 2009	NO_QUERY	Anthony_Nguyen	Bed. Class 8-12. Work 12-3. Gym 3-5 or 6. Then class 6-10. Another day that's gonna fly by. I miss my girlfriend
40	0	1467818020	Mon Apr 06 22:21:39 PDT 2009	NO_QUERY	itsanimesh	really don't feel like getting up today... but got to study to for tomorrows practical exam...
41	0	1467818481	Mon Apr 06 22:21:46 PDT 2009	NO_QUERY	lionslamb	He's the reason for the teardrops on my guitar the only one who has enough of me to break my heart
42	0	1467818603	Mon Apr 06 22:21:49 PDT 2009	NO_QUERY	kennypham	Sad, sad, sad. I don't know why but I hate this feeling I wanna sleep and I still can't!
43	0	1467818900	Mon Apr 06 22:21:53 PDT 2009	NO_QUERY	DdubsShellBell	@JonathanRKnight Awww I soo wish I was there to see you finally comfortable! Im sad that I missed it
44	0	1467819022	Mon Apr 06 22:21:56 PDT 2009	NO_QUERY	hpfangirl94	Falling asleep. Just heard about that Tracy girl's body being found. How sad My heart breaks for that family.
45	0	1467819650	Mon Apr 06 22:22:05 PDT 2009	NO_QUERY	antzpantz	@Viennah Yay! I'm happy for you with your job! But that also means less time for me and you...
46	0	1467819712	Mon Apr 06 22:22:06 PDT 2009	NO_QUERY	labrt2004	Just checked my user timeline on my blackberry, it looks like the twanking is still happening Are ppl still having probs w/ BGs and UIDs?
47	0	1467819812	Mon Apr 06 22:22:07 PDT 2009	NO_QUERY	IrisJumbe	Oh man...was ironing @jeancjumbe's fave top to wear to a meeting. Burnt it
48	0	1467820206	Mon Apr 06 22:22:13 PDT 2009	NO_QUERY	peacoats	is strangely sad about LiLo and SamRo breaking up.
49	0	1467820835	Mon Apr 06 22:22:25 PDT 2009	NO_QUERY	cyantist	@tea oh! i'm so sorry i didn't think about that before retweeting.
50	0	1467820863	Mon Apr 06 22:22:23 PDT 2009	NO_QUERY	tautao	Broadband plan 'a massive broken promise' http://tinyurl.com/dcuc33 via www.diigo.com/~tautao Still waiting for broadband we are
51	0	1467820906	Mon Apr 06 22:22:24 PDT 2009	NO_QUERY	voyage2k	@localtweeps Wow, tons of replies from you, may have to unfollow so I can see my friends' tweets, you're scrolling the feed a lot.
52	0	1467821085	Mon Apr 06 22:22:26 PDT 2009	NO_QUERY	crzy_cdn_bulas	our duck and chicken are taking wayyy too long to hatch
53	0	1467821338	Mon Apr 06 22:22:30 PDT 2009	NO_QUERY	justnetgirl	Put vacation photos online a few yrs ago. PC crashed, and now I forget the name of the site.
54	0	1467821455	Mon Apr 06 22:22:32 PDT 2009	NO_QUERY	CiaraRenee	I need a hug
55	0	1467821715	Mon Apr 06 22:22:37 PDT 2009	NO_QUERY	deelau	@andywana Not sure what they are, only that they are PoS! As much as I want to, I dont think can trade away company assets sorry andy!
56	0	1467822384	Mon Apr 06 22:22:47 PDT 2009	NO_QUERY	Lindsey0920	@oanhLove I hate when that happens...
57	0	1467822389	Mon Apr 06 22:22:47 PDT 2009	NO_QUERY	HybridMink	I have a sad feeling that Dallas is not going to show up I gotta say though, you'd think more shows would use music from the game. mmm
58	0	1467822519	Mon Apr 06 22:22:49 PDT 2009	NO_QUERY	gzacher	Ugh....92 degrees tomorrow
59	0	1467822522	Mon Apr 06 22:22:49 PDT 2009	NO_QUERY	Jenn_L	Where did u move to? I thought u were already in sd. ?? Hmmm. Random u found me. Glad to hear yer doing well.
60	0	1467822687	Mon Apr 06 22:22:52 PDT 2009	NO_QUERY	xVivaLaJuicyx	@BatManYNG I miss my ps3, it's out of commission Wutcha playing? Have you copped 'Blood On The Sand'?
61	0	1467822918	Mon Apr 06 22:22:55 PDT 2009	NO_QUERY	krbleyle	just leaving the parking lot of work!
62	0	1467823437	Mon Apr 06 22:23:03 PDT 2009	NO_QUERY	xpika	The Life is cool. But not for Me.
63	0	1467823770	Mon Apr 06 22:23:08 PDT 2009	NO_QUERY	Henkuyinepu	Sadly though, I've never gotten to experience the post coitus cigarette before, and now I never will.
64	0	1467823851	Mon Apr 06 22:23:09 PDT 2009	NO_QUERY	ericg622	I had such a nice day. Too bad the rain comes in tomorrow at 5am
65	0	1467824199	Mon Apr 06 22:23:15 PDT 2009	NO_QUERY	adri_mane	@Starrbby too bad I won't be around I lost my job and can't even pay my phone bill lmao aw shucks
66	0	1467824664	Mon Apr 06 22:23:23 PDT 2009	NO_QUERY	a_mariepyt	Damm back to school tomorrow
67	0	1467824967	Mon Apr 06 22:23:28 PDT 2009	NO_QUERY	playboybacon	Mo jobs, no money. how in the hell is min wage here 4 f'n clams an hour?
68	0	1467825003	Mon Apr 06 22:23:28 PDT 2009	NO_QUERY	leslierosales	@katortiz Not forever... See you soon!
69	0	1467825084	Mon Apr 06 22:23:30 PDT 2009	NO_QUERY	PresidentSnow	@Lt_Algonquin agreed, I saw the failwhale allllll day today.
70	0	1467825411	Mon Apr 06 22:23:35 PDT 2009	NO_QUERY	michrod	@jdarter Oh! Haha... dude I dont really look at em unless someone says HEY I ADDED YOU. Sorry I'm so terrible at that. I need a pop up!
71	0	1467825642	Mon Apr 06 22:23:39 PDT 2009	NO_QUERY	timmelko	@ninjen I'm sure you're right... I need to start working out with you and the Nikster... Or Jared at least!
72	0	1467825863	Mon Apr 06 22:23:43 PDT 2009	NO_QUERY	BrookeAmanda	i really hate how people diss my bands! Trace is clearly NOT ugly!
73	0	1467825883	Mon Apr 06 22:23:43 PDT 2009	NO_QUERY	deelau	Gym attire today was: Puma singlet, Adidas shorts.......and black business socks and leather shoes Lucky did not run into any cute girls.
74	0	1467826052	Mon Apr 06 22:23:45 PDT 2009	NO_QUERY	paulseverio	Why won't you show my location?! http://twitpic.com/2y2es
75	0	1467833672	Mon Apr 06 22:25:44 PDT 2009	NO_QUERY	iv3tte	No picnic my phone smells like citrus.
76	0	1467833690	Mon Apr 06 22:25:44 PDT 2009	NO_QUERY	fedunska	@ashleyac My donkey is sensitive about such comments. Nevertheless, he'd (and me'd) be glad to see your mug asap. Charger is still awol.
77	0	1467833736	Mon Apr 06 22:25:45 PDT 2009	NO_QUERY	MagicalMason	No new csi tonight. FML
78	0	1467833799	Mon Apr 06 22:25:46 PDT 2009	NO_QUERY	kaelaaa	i think my arms are sore from tennis
79	0	1467834001	Mon Apr 06 22:25:49 PDT 2009	NO_QUERY	emo_holic	wonders why someone that u like so much can make you so unhappy in a split seccond . depressed .
80	0	1467834053	Mon Apr 06 22:25:52 PDT 2009	NO_QUERY	thelazyboy	sleep soon... i just hate saying bye and see you tomorrow for the night.
81	0	1467834227	Mon Apr 06 22:25:53 PDT 2009	NO_QUERY	driveaway2008	@statravelAU just got ur newsletter, those fares really are unbelievable, shame I already booked and paid for mine
82	0	1467834239	Mon Apr 06 22:25:53 PDT 2009	NO_QUERY	mscha	missin' the boo
83	0	1467834265	Mon Apr 06 22:25:54 PDT 2009	NO_QUERY	mike_webster_au	@markhardy1974 Me too #itm
84	0	1467834284	Mon Apr 06 22:25:54 PDT 2009	NO_QUERY	basiabeans	Damn... I don't have any chalk! MY CHALKBOARD IS USELESS
85	0	1467834400	Mon Apr 06 22:25:56 PDT 2009	NO_QUERY	calihonda2001	had a blast at the Getty Villa, but hates that she's had a sore throat all day. It's just getting worse too
86	0	1467834817	Mon Apr 06 22:26:02 PDT 2009	NO_QUERY	djwayneski	@msdrama hey missed ya at the meeting sup mama
87	0	1467835085	Mon Apr 06 22:26:06 PDT 2009	NO_QUERY	Ceejison	My tummy hurts. I wonder if the hypnosis has anything to do with it? If so, it's working, I get it, STOP SMOKING!!!
88	0	1467835198	Mon Apr 06 22:26:08 PDT 2009	NO_QUERY	ItsBrigittaYo	why is it always the fat ones?!
89	0	1467835305	Mon Apr 06 22:26:10 PDT 2009	NO_QUERY	MissLaura317	@januarycrimson Sorry, babe!! My fam annoys me too. Thankfully, they're asleep right now. Muahaha. *evil laugh*
90	0	1467835345	Mon Apr 06 22:26:10 PDT 2009	NO_QUERY	RU_it_girl	@Hollywoodheat I should have paid more attention when we covered photoshop in my webpage design class in undergrad
91	0	1467835577	Mon Apr 06 22:26:14 PDT 2009	NO_QUERY	viviana09	wednesday my b-day! don't know what 2 do!!
92	0	1467835880	Mon Apr 06 22:26:18 PDT 2009	NO_QUERY	disneyfan4eva	Poor cameron (the hills)
93	0	1467836024	Mon Apr 06 22:26:21 PDT 2009	NO_QUERY	RoseMaryK	pray for me please, the ex is threatening to start sh** at my/our babies 1st Birthday party. what a jerk. and I still have a headache
94	0	1467836111	Mon Apr 06 22:26:22 PDT 2009	NO_QUERY	perrohunter	@makeherfamous hmm , do u really enjoy being with him ? if the problems are too constants u should think things more , find someone ulike
95	0	1467836448	Mon Apr 06 22:26:27 PDT 2009	NO_QUERY	Dogbook	Strider is a sick little puppy http://apps.facebook.com/dogbook/profile/view/5248435
96	0	1467836500	Mon Apr 06 22:26:28 PDT 2009	NO_QUERY	natalieantipas	so rylee,grace...wana go steve's party or not?? SADLY SINCE ITS EASTER I WNT B ABLE 2 DO MUCH BUT OHH WELL.....
97	0	1467836576	Mon Apr 06 22:26:29 PDT 2009	NO_QUERY	timdonnelly	hey, I actually won one of my bracket pools! Too bad it wasn't the one for money
98	0	1467836583	Mon Apr 06 22:26:29 PDT 2009	NO_QUERY	homeworld	@stark YOU don't follow me, either and i work for you!
99	0	1467836859	Mon Apr 06 22:26:33 PDT 2009	NO_QUERY	willy_chaz	A bad nite for the favorite teams: Astros and Spartans lose. The nite out with T.W. was good.
100	0	1467836873	Mon Apr 06 22:26:33 PDT 2009	NO_QUERY	LeakySpoon	Body Of Missing Northern Calif. Girl Found: Police have found the remains of a missing Northern California girl .. http://tr.im/imji
Similar Datasets
Google Play Store Apps
Google Play Store Apps
Trending YouTube Video Statistics
Trending YouTube Video Statistics
New York City Airbnb Open Data
New York City Airbnb Open Data
Chest X-Ray Images (Pneumonia)
Chest X-Ray Images (Pneumonia)
Amazon Fine Food Reviews
Amazon Fine Food Reviews
134,261 views
23,171 downloads
57 kernels
4 topics
View more activity
© 2020 Kaggle Inc
Our Team Terms Privacy Contact/Support


'''


def _create_frequency_table(text_string) -> dict:
    """
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.
    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    :rtype: dict
    """
    stopWords = set(stopwords.words("english"))
    # adding beautiful soap logic to get clean text from html 
    
    words = word_tokenize(review_text)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _score_sentences(sentences, freqTable) -> dict:
    """
    score a sentence by its words
    Basic algorithm: adding the frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        word_count_in_sentence_except_stop_words = 0
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                word_count_in_sentence_except_stop_words += 1
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        if sentence[:10] in sentenceValue:
            sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] / word_count_in_sentence_except_stop_words

        '''
        Notice that a potential issue with our score algorithm is that long sentences will have an advantage over short sentences. 
        To solve this, we're dividing every sentence score by the number of words in the sentence.
        
        Note that here sentence[:10] is the first 10 character of any sentence, this is to save memory while saving keys of
        the dictionary.
        '''

    return sentenceValue


def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    average = (sumValues / len(sentenceValue))

    return average


def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary




def run_summarization(text):
    
    
    
    # 1 Create the word frequency table
    freq_table = _create_frequency_table(text)

    '''
    We already have a sentence tokenizer, so we just need 
    to run the sent_tokenize() method to create the array of sentences.
    '''

    # 2 Tokenize the sentences
    sentences = sent_tokenize(text)

    # 3 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(sentences, freq_table)

    # 4 Find the threshold
    threshold = _find_average_score(sentence_scores)

    # 5 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, 1.3 * threshold)

    return summary


if __name__ == '__main__':
    #1.Remove HTML
    review_text = BeautifulSoup(text_str).get_text()
    result = run_summarization(review_text)
    print(result)
