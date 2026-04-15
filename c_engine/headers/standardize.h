#ifndef STANDARDIZE_H
#define STANDARDIZE_H

#include <string.h>
#include <ctype.h>
#include <stdio.h>

typedef struct {
    const char *contraction;
    const char *expansion;
} ContractionRule;

static const ContractionRule contractions[] = {
    {"i'm",       "i am"},
    {"i've",      "i have"},
    {"i'll",      "i will"},
    {"i'd",       "i would"},
    {"you're",    "you are"},
    {"you've",    "you have"},
    {"you'll",    "you will"},
    {"you'd",     "you would"},
    {"he's",      "he is"},
    {"she's",     "she is"},
    {"it's",      "it is"},
    {"we're",     "we are"},
    {"we've",     "we have"},
    {"we'll",     "we will"},
    {"we'd",      "we would"},
    {"they're",   "they are"},
    {"they've",   "they have"},
    {"they'll",   "they will"},
    {"they'd",    "they would"},
    {"that's",    "that is"},
    {"who's",     "who is"},
    {"what's",    "what is"},
    {"where's",   "where is"},
    {"when's",    "when is"},
    {"why's",     "why is"},
    {"how's",     "how is"},
    {"isn't",     "is not"},
    {"aren't",    "are not"},
    {"wasn't",    "was not"},
    {"weren't",   "were not"},
    {"hasn't",    "has not"},
    {"haven't",   "have not"},
    {"hadn't",    "had not"},
    {"doesn't",   "does not"},
    {"don't",     "do not"},
    {"didn't",    "did not"},
    {"won't",     "will not"},
    {"wouldn't",  "would not"},
    {"shouldn't", "should not"},
    {"couldn't",  "could not"},
    {"can't",     "cannot"},
    {"let's",     "let us"},
    {NULL, NULL}
};

static void normalize_unicode(char *text) {
    char *r = text, *w = text;
    while (*r) {
        unsigned char b = (unsigned char)*r;

        if (b == 0xE2 && (unsigned char)r[1] == 0x80) {
            unsigned char b2 = (unsigned char)r[2];
            if (b2 >= 0x8B && b2 <= 0x8D) { r += 3; continue; }          // zero-width
            if (b2 == 0x93 || b2 == 0x94) { *w++ = '-'; r += 3; continue; } // em/en dash
            if (b2 == 0x9C || b2 == 0x9D) { *w++ = '"'; r += 3; continue; } // curly quotes
            if (b2 == 0x98 || b2 == 0x99) { *w++ = '\''; r += 3; continue; }
            if (b2 == 0xA6) { *w++ = '.'; *w++ = '.'; *w++ = '.'; r += 3; continue; } // ellipsis
        }

        if (b == 0xC2 && (unsigned char)r[1] == 0xA0) { *w++ = ' '; r += 2; continue; } // nbsp

        if (b == 0xEF && (unsigned char)r[1] == 0xB8 && (unsigned char)r[2] == 0x8F) { r += 3; continue; } // variation selector

        if (b > 0x7F) {
            int skip = 1;
            if ((b & 0xE0) == 0xC0) skip = 2;
            else if ((b & 0xF0) == 0xE0) skip = 3;
            else if ((b & 0xF8) == 0xF0) skip = 4;
            r += skip;
            *w++ = ' ';
            continue;
        }

        *w++ = *r++;
    }
    *w = '\0';
}

#define EXPAND_BUF_SIZE 262144

static void expand_contractions(char *text) {
    static char tmp[EXPAND_BUF_SIZE];
    char *r = text, *w = tmp, *end = tmp + EXPAND_BUF_SIZE - 64;

    while (*r && w < end) {
        int replaced = 0;
        for (int i = 0; contractions[i].contraction; i++) {
            size_t clen = strlen(contractions[i].contraction);
            int match = 1;
            for (size_t j = 0; j < clen; j++) {
                if (tolower((unsigned char)r[j]) != (unsigned char)contractions[i].contraction[j]) {
                    match = 0; break;
                }
            }
            if (match) {
                char next = r[clen];
                if (next == '\0' || !isalpha((unsigned char)next)) {
                    size_t elen = strlen(contractions[i].expansion);
                    memcpy(w, contractions[i].expansion, elen);
                    w += elen;
                    r += clen;
                    replaced = 1;
                    break;
                }
            }
        }
        if (!replaced) *w++ = *r++;
    }
    *w = '\0';
    strcpy(text, tmp);
}

static void standardize_text(char *text) {
    static int firstCall = 1;
    if (firstCall) {
        printf(" [standardize] engaged\n");
        firstCall = 0;
    }

    for (char *p = text; *p; p++)
        *p = (char)tolower((unsigned char)*p);

    normalize_unicode(text);
    expand_contractions(text);
}

#endif
