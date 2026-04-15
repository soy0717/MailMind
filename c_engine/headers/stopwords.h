#ifndef STOPWORDS_H
#define STOPWORDS_H

#include <string.h>
#include <stdio.h>

static const char *STOPWORDS[] = {
    "a","about","above","after","again","against","all","am","an","and",
    "any","are","as","at","be","because","been","before","being","below",
    "between","both","but","by","can","could","did","do","does","doing",
    "down","during","each","few","for","from","further","get","got","had",
    "has","have","having","he","her","here","hers","herself","him",
    "himself","his","how","if","in","into","is","it","its","itself",
    "just","know","let","like","ll","may","me","might","more","most",
    "much","must","my","myself","need","no","nor","not","now","of","off",
    "on","once","only","or","other","our","ours","ourselves","out","over",
    "own","re","really","same","say","shall","she","should","so","some",
    "still","such","take","than","that","the","their","theirs","them",
    "themselves","then","there","these","they","this","those","through",
    "to","too","under","until","up","us","ve","very","want","was","we",
    "well","were","what","when","where","which","while","who","whom",
    "why","will","with","would","you","your","yours","yourself",
    "yourselves"
};

static const int STOPWORD_COUNT = sizeof(STOPWORDS) / sizeof(STOPWORDS[0]);

static int is_stopword(const char *word) {
    static int firstCall = 1;
    if (firstCall) {
        printf(" [stopwords] engaged\n");
        firstCall = 0;
    }

    int lo = 0, hi = STOPWORD_COUNT - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int cmp = strcmp(word, STOPWORDS[mid]);
        if (cmp == 0) return 1;
        else if (cmp < 0) hi = mid - 1;
        else lo = mid + 1;
    }
    return 0;
}

#endif