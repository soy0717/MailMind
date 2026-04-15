#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "headers/segment.h"
#include "headers/standardize.h"
#include "headers/denoise.h"
#include "headers/tokenize.h"
#include "headers/stopwords.h"
#include "headers/lemmatize.h"

#define MAX_LINE    524288
#define MAX_TEXT    524288
#define MAX_OUTPUT  262144

static void escape_csv_field(const char *src, char *out, int maxLen) {
    int i = 0;
    out[i++] = '"';
    for (const char *p = src; *p && i < maxLen - 3; p++) {
        if (*p == '"') {
            out[i++] = '"';
            out[i++] = '"';
        } else {
            out[i++] = *p;
        }
    }
    out[i++] = '"';
    out[i] = '\0';
}

static void process_email(const char *rawText,
                          char *sentenceOut, int sentMax,
                          char *cleanOut,   int cleanMax,
                          char *lemmaOut,   int lemmaMax)
{
    static char buf[MAX_TEXT];
    strncpy(buf, rawText, MAX_TEXT - 1);
    buf[MAX_TEXT - 1] = '\0';

    segment_content(buf);
    standardize_text(buf);
    remove_noise(buf);

    strncpy(sentenceOut, buf, sentMax - 1);
    sentenceOut[sentMax - 1] = '\0';

    static TokenList tokens;
    tokenize_text(buf, &tokens);

    static TokenList filtered;
    filtered.word_count = 0;
    for (int i = 0; i < tokens.word_count; i++) {
        if (!is_stopword(tokens.words[i])) {
            strncpy(filtered.words[filtered.word_count], tokens.words[i], MAX_TOKEN_LEN - 1);
            filtered.words[filtered.word_count][MAX_TOKEN_LEN - 1] = '\0';
            filtered.word_count++;
            if (filtered.word_count >= MAX_TOKENS) break;
        }
    }

    int pos = 0;
    for (int i = 0; i < filtered.word_count && pos < cleanMax - MAX_TOKEN_LEN; i++) {
        if (i > 0) cleanOut[pos++] = ' ';
        int tlen = (int)strlen(filtered.words[i]);
        memcpy(cleanOut + pos, filtered.words[i], tlen);
        pos += tlen;
    }
    cleanOut[pos] = '\0';

    static TokenList lemmatized;
    lemmatized.word_count = 0;
    for (int i = 0; i < filtered.word_count; i++) {
        char lemma[MAX_TOKEN_LEN];
        lemmatize_word(filtered.words[i], lemma, MAX_TOKEN_LEN);
        strncpy(lemmatized.words[lemmatized.word_count], lemma, MAX_TOKEN_LEN - 1);
        lemmatized.words[lemmatized.word_count][MAX_TOKEN_LEN - 1] = '\0';
        lemmatized.word_count++;
        if (lemmatized.word_count >= MAX_TOKENS) break;
    }

    pos = 0;
    for (int i = 0; i < lemmatized.word_count && pos < lemmaMax - MAX_TOKEN_LEN; i++) {
        if (i > 0) lemmaOut[pos++] = ' ';
        int tlen = (int)strlen(lemmatized.words[i]);
        memcpy(lemmaOut + pos, lemmatized.words[i], tlen);
        pos += tlen;
    }
    lemmaOut[pos] = '\0';
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: preprocess.exe <input.tsv> <output.csv>\n");
        return 1;
    }

    fprintf(outFile, "email_id,sentence_text,clean_text,lemmas,trueDomain\n");

    char *lineBuffer = (char *)malloc(MAX_LINE);
    if (!lineBuffer) {
        fprintf(stderr, "Error: out of memory\n");
        fclose(inFile); fclose(outFile);
        return 1;
    }

    static char sentenceBuf[MAX_OUTPUT];
    static char cleanBuf[MAX_OUTPUT];
    static char lemmaBuf[MAX_OUTPUT];
    static char csvSentence[MAX_OUTPUT + 64];
    static char csvClean[MAX_OUTPUT + 64];
    static char csvLemmas[MAX_OUTPUT + 64];

    int emailCount = 0;
    int skipped    = 0;

    printf("[preprocessor] Starting up\n");
    printf("[preprocessor] Reading from %s\n", argv[1]);

    while (fgets(lineBuffer, MAX_LINE, inFile)) {
        int len = (int)strlen(lineBuffer);
        while (len > 0 && (lineBuffer[len-1] == '\n' || lineBuffer[len-1] == '\r'))
            lineBuffer[--len] = '\0';

        if (len == 0) continue;

        char *emailId = lineBuffer;
        char *tab1 = strchr(lineBuffer, '\t');
        if (!tab1) { skipped++; continue; }
        *tab1 = '\0';

        char *domain = tab1 + 1;
        char *tab2 = strchr(domain, '\t');
        if (!tab2) { skipped++; continue; }
        *tab2 = '\0';

        char *text = tab2 + 1;

        process_email(text,
                      sentenceBuf, MAX_OUTPUT,
                      cleanBuf,    MAX_OUTPUT,
                      lemmaBuf,    MAX_OUTPUT);

        escape_csv_field(sentenceBuf, csvSentence, MAX_OUTPUT + 64);
        escape_csv_field(cleanBuf,    csvClean,    MAX_OUTPUT + 64);
        escape_csv_field(lemmaBuf,    csvLemmas,   MAX_OUTPUT + 64);

        fprintf(outFile, "%s,%s,%s,%s,%s\n",
                emailId, csvSentence, csvClean, csvLemmas, domain);

        emailCount++;
        if (emailCount % 500 == 0)
            printf("[preprocessor] Processed %d emails\n", emailCount);
    }

    printf("[preprocessor] Total processed: %d, Skipped: %d\n", emailCount, skipped);
    printf("[preprocessor] Output: %s\n", argv[2]);

    free(lineBuffer);
    fclose(inFile);
    fclose(outFile);
    return 0;
}
