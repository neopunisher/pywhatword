from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import sys
import urllib2

def extract_words(text):
    '''
    here we are extracting features to use in our classifier. We want to pull all the words in our input
    porterstem them and grab the most significant bigrams to add to the mix as well.
    '''

    stemmer = PorterStemmer()

    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(text)

    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 500)

    for bigram_tuple in bigrams:
        x = "%s %s" % bigram_tuple
        tokens.append(x)

    result =  [stemmer.stem(x.lower()) for x in tokens if x not in stopwords.words('english') and len(x) > 1]
    return result

def get_feature(word):
    return dict([(word, True)])


def bag_of_words(words):
    return dict([(word, True) for word in words])


def create_training_dict(text, sense):
    ''' returns a dict ready for a classifier's test method '''
    tokens = extract_words(text)
    return [(bag_of_words(tokens), sense)]



def run_classifier_tests(classifier):
    testfiles = [{'fruit': 'http://litfuel.net/plush/files/disambiguation/apple-fruit-training.txt'},
                 {'company': 'http://litfuel.net/plush/files/disambiguation/apple-company-training.txt'}]
    testfeats = []
    for file in testfiles:
        for sense, loc in file.iteritems():
            for line in urllib2.urlopen(loc):
                testfeats = testfeats + create_training_dict(line, sense)


    acc = accuracy(classifier, testfeats) * 100
    print 'accuracy: %.2f%%' % acc

    sys.exit()


if __name__ == '__main__':

    # create our dict of training data
    texts = {}
    texts['fruit'] = 'http://litfuel.net/plush/files/disambiguation/apple-fruit.txt'
    texts['company'] = 'http://litfuel.net/plush/files/disambiguation/apple-company.txt'

    #holds a dict of features for training our classifier
    train_set = []

    # loop through each item, grab the text, tokenize it and create a training feature with it
    for sense, file in texts.iteritems():
        print "training %s " % sense
        text = urllib2.urlopen(file, 'r').read()
        features = extract_words(text)
        train_set = train_set + [(get_feature(word), sense) for word in features]


    classifier = NaiveBayesClassifier.train(train_set)

    # uncomment out this line to see the most informative words the classifier will use
    #classifier.show_most_informative_features(20)


    # uncomment out this line to see how well our accuracy is using some hand curated tweets
    #run_classifier_tests(classifier)


    for line in urllib2.urlopen("http://litfuel.net/plush/files/disambiguation/apple-tweets.txt", 'r'):

        tokens = bag_of_words(extract_words(line))
        decision = classifier.classify(tokens)
        result = "%s - %s" % (decision,line )
        print result