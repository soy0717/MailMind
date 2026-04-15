#ifndef SEGMENT_H
#define SEGMENT_H

#include <string.h>
#include <ctype.h>
#include <stdio.h>

static const char *sigMarkers[] = {
    "best regards", "kind regards", "warm regards",
    "thanks and regards", "regards,", "sincerely,",
    "cheers,", "thanks,", "thank you,", "yours truly,",
    "sent from my iphone", "sent from my android",
    "get outlook for", "get the new linkedin",
    "--\n", "-- \n", "___", NULL
};

static const char *fwdMarkers[] = {
    "---------- forwarded message",
    "---original message---",
    "--- original message ---",
    "begin forwarded message",
    NULL
};

static void strip_reply_lines(char *text) {
    char *r = text, *w = text;
    int atLineStart = 1;
    while (*r) {
        if (atLineStart && *r == '>') {
            while (*r && *r != '\n') r++;
            if (*r == '\n') r++;
            atLineStart = 1;
            continue;
        }
        atLineStart = (*r == '\n');
        *w++ = *r++;
    }
    *w = '\0';
}

static const char *find_ci(const char *haystack, const char *needle) {
    size_t nlen = strlen(needle);
    if (!nlen) return haystack;
    for (const char *p = haystack; *p; p++) {
        int ok = 1;
        for (size_t i = 0; i < nlen; i++) {
            if (tolower((unsigned char)p[i]) != tolower((unsigned char)needle[i])) {
                ok = 0; break;
            }
        }
        if (ok) return p;
    }
    return NULL;
}

static void segment_content(char *text) {
    static int firstCall = 1;
    if (firstCall) {
        printf(" [segment] engaged\n");
        firstCall = 0;
    }

    strip_reply_lines(text);

    const char *cutAt = NULL;

    for (int i = 0; fwdMarkers[i]; i++) {
        const char *hit = find_ci(text, fwdMarkers[i]);
        if (hit && (!cutAt || hit < cutAt)) cutAt = hit;
    }

    for (int i = 0; sigMarkers[i]; i++) {
        const char *hit = find_ci(text, sigMarkers[i]);
        if (hit && (!cutAt || hit < cutAt)) cutAt = hit;
    }

    if (cutAt) *(char *)cutAt = '\0';
}

#endif
