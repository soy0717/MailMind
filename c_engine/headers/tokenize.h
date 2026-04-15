#ifndef TOKENIZE_H
#define TOKENIZE_H

#include <string.h>
#include <ctype.h>
#include <stdio.h>

#define MAX_TOKENS    2048
#define MAX_TOKEN_LEN 64

typedef struct {
    char words[MAX_TOKENS][MAX_TOKEN_LEN];
    int  word_count;
} TokenList;

static void tokenize_text(const char *text, TokenList *out) {
    static int firstCall = 1;
    if (firstCall) {
        printf(" [tokenize] engaged\n");
        firstCall = 0;
    }

    out->word_count = 0;
    const char *p = text;

    while (*p && out->word_count < MAX_TOKENS) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        char word[MAX_TOKEN_LEN];
        int wlen = 0;
        while (*p && !isspace((unsigned char)*p) && wlen < MAX_TOKEN_LEN - 1)
            word[wlen++] = *p++;
        word[wlen] = '\0';

        int start = 0;
        while (start < wlen && (word[start]=='.' || word[start]==',' ||
               word[start]=='!' || word[start]=='?' || word[start]==';' || word[start]==':'))
            start++;

        int end = wlen - 1;
        while (end > start && (word[end]=='.' || word[end]==',' ||
               word[end]=='!' || word[end]=='?' || word[end]==';' || word[end]==':'))
            end--;

        int cleanLen = end - start + 1;
        if (cleanLen <= 1) continue;

        int allDigits = 1;
        for (int i = start; i <= end; i++) {
            if (!isdigit((unsigned char)word[i])) { allDigits = 0; break; }
        }
        if (allDigits) continue;

        memcpy(out->words[out->word_count], word + start, cleanLen);
        out->words[out->word_count][cleanLen] = '\0';
        out->word_count++;
    }
}

#endif
