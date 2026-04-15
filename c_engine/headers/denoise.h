#ifndef DENOISE_H
#define DENOISE_H

#include <string.h>
#include <ctype.h>
#include <stdio.h>

static const char *boilerplateList[] = {
    "this email was intended for",
    "learn why we included this",
    "you are receiving",
    "unsubscribe",
    "manage job alerts",
    "view in browser",
    "linkedin corporation",
    "west maude avenue",
    "sunnyvale, ca",
    "linkedin and the linkedin logo",
    "registered trademarks",
    "also available on mobile",
    "get the new linkedin desktop app",
    "try premium",
    "1-month free trial",
    "cancel anytime",
    "edit alert",
    "see all jobs",
    "manage alerts",
    "top applicant",
    "increase your chances",
    NULL
};

static void strip_html_tags(char *text) {
    char *r = text, *w = text;
    int inTag = 0;
    while (*r) {
        if (*r == '<')      { inTag = 1; r++; continue; }
        if (*r == '>')      { inTag = 0; *w++ = ' '; r++; continue; }
        if (!inTag) *w++ = *r;
        r++;
    }
    *w = '\0';
}

static void strip_urls(char *text) {
    char *r = text, *w = text;
    while (*r) {
        if (r[0]=='h' && r[1]=='t' && r[2]=='t' && r[3]=='p' &&
            (r[4]==':' || (r[4]=='s' && r[5]==':'))) {
            while (*r && !isspace((unsigned char)*r) && *r!='>' && *r!=')' && *r!=']')
                r++;
            *w++ = ' ';
            continue;
        }
        // www.
        if (r[0]=='w' && r[1]=='w' && r[2]=='w' && r[3]=='.') {
            while (*r && !isspace((unsigned char)*r) && *r!='>' && *r!=')' && *r!=']')
                r++;
            *w++ = ' ';
            continue;
        }
        *w++ = *r++;
    }
    *w = '\0';
}

static void strip_email_addresses(char *text) {
    char *r = text, *w = text;
    while (*r) {
        if (!isspace((unsigned char)*r)) {
            char *wordStart = r;
            int hasAt = 0;
            while (*r && !isspace((unsigned char)*r)) {
                if (*r == '@') hasAt = 1;
                r++;
            }
            if (!hasAt) {
                while (wordStart < r) *w++ = *wordStart++;
            } else {
                *w++ = ' ';
            }
        } else {
            *w++ = *r++;
        }
    }
    *w = '\0';
}

static void strip_boilerplate(char *text) {
    char *lineStart = text, *w = text;
    while (*lineStart) {
        char *lineEnd = lineStart;
        while (*lineEnd && *lineEnd != '\n') lineEnd++;

        int isBad = 0;
        for (int i = 0; boilerplateList[i]; i++) {
            const char *phrase = boilerplateList[i];
            size_t plen = strlen(phrase);
            for (const char *p = lineStart; p + plen <= lineEnd; p++) {
                int match = 1;
                for (size_t j = 0; j < plen; j++) {
                    if (tolower((unsigned char)p[j]) != tolower((unsigned char)phrase[j])) {
                        match = 0; break;
                    }
                }
                if (match) { isBad = 1; break; }
            }
            if (isBad) break;
        }

        if (!isBad) {
            while (lineStart < lineEnd) *w++ = *lineStart++;
            if (*lineEnd == '\n') *w++ = *lineEnd;
        }

        if (*lineEnd == '\n') lineEnd++;
        lineStart = lineEnd;
    }
    *w = '\0';
}

static void collapse_whitespace(char *text) {
    char *r = text, *w = text;
    int prevSpace = 1;
    while (*r) {
        if (isspace((unsigned char)*r)) {
            if (!prevSpace) { *w++ = ' '; prevSpace = 1; }
            r++;
        } else {
            *w++ = *r++;
            prevSpace = 0;
        }
    }
    if (w > text && *(w-1) == ' ') w--;
    *w = '\0';
}

static void strip_special_chars(char *text) {
    char *r = text, *w = text;
    while (*r) {
        char c = *r;
        if (isalnum((unsigned char)c) || c==' ' || c=='.' || c==',' ||
            c=='!' || c=='?' || c==';' || c==':' || c=='\'' || c=='-' || c=='\n')
            *w++ = c;
        else
            *w++ = ' ';
        r++;
    }
    *w = '\0';
}

static void remove_noise(char *text) {
    static int firstCall = 1;
    if (firstCall) {
        printf(" [denoise] engaged\n");
        firstCall = 0;
    }

    strip_html_tags(text);
    strip_urls(text);
    strip_email_addresses(text);
    strip_boilerplate(text);
    strip_special_chars(text);
    collapse_whitespace(text);
}

#endif
