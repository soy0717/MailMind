#ifndef LEMMATIZE_H
#define LEMMATIZE_H

#include <string.h>
#include <ctype.h>
#include <stdio.h>

typedef struct {
    const char *word;
    const char *lemma;
} LemmaEntry;

static const LemmaEntry irregulars[] = {
    {"am","be"},{"is","be"},{"are","be"},{"was","be"},{"were","be"},
    {"been","be"},{"being","be"},
    {"has","have"},{"had","have"},{"having","have"},
    {"does","do"},{"did","do"},{"doing","do"},{"done","do"},
    {"goes","go"},{"went","go"},{"gone","go"},{"going","go"},
    {"said","say"},{"says","say"},{"saying","say"},
    {"made","make"},{"makes","make"},{"making","make"},
    {"took","take"},{"taken","take"},{"takes","take"},{"taking","take"},
    {"came","come"},{"comes","come"},{"coming","come"},
    {"saw","see"},{"seen","see"},{"sees","see"},{"seeing","see"},
    {"knew","know"},{"known","know"},{"knows","know"},{"knowing","know"},
    {"got","get"},{"gets","get"},{"gotten","get"},{"getting","get"},
    {"gave","give"},{"given","give"},{"gives","give"},{"giving","give"},
    {"found","find"},{"finds","find"},{"finding","find"},
    {"thought","think"},{"thinks","think"},{"thinking","think"},
    {"told","tell"},{"tells","tell"},{"telling","tell"},
    {"became","become"},{"becomes","become"},{"becoming","become"},
    {"left","leave"},{"leaves","leave"},{"leaving","leave"},
    {"felt","feel"},{"feels","feel"},{"feeling","feel"},
    {"puts","put"},{"putting","put"},
    {"brought","bring"},{"brings","bring"},{"bringing","bring"},
    {"began","begin"},{"begun","begin"},{"begins","begin"},{"beginning","begin"},
    {"kept","keep"},{"keeps","keep"},{"keeping","keep"},
    {"wrote","write"},{"written","write"},{"writes","write"},{"writing","write"},
    {"stood","stand"},{"stands","stand"},{"standing","stand"},
    {"ran","run"},{"runs","run"},{"running","run"},
    {"sat","sit"},{"sits","sit"},{"sitting","sit"},
    {"sent","send"},{"sends","send"},{"sending","send"},
    {"built","build"},{"builds","build"},{"building","build"},
    {"understood","understand"},{"understands","understand"},
    {"learnt","learn"},{"learned","learn"},{"learns","learn"},{"learning","learn"},
    {"grew","grow"},{"grown","grow"},{"grows","grow"},{"growing","grow"},
    {"paid","pay"},{"pays","pay"},{"paying","pay"},
    {"met","meet"},{"meets","meet"},{"meeting","meet"},
    {"led","lead"},{"leads","lead"},{"leading","lead"},
    {"sets","set"},{"setting","set"},
    {"reads","read"},{"reading","read"},
    {"spent","spend"},{"spends","spend"},{"spending","spend"},
    {"won","win"},{"wins","win"},{"winning","win"},
    {"held","hold"},{"holds","hold"},{"holding","hold"},
    {"sold","sell"},{"sells","sell"},{"selling","sell"},
    {"chose","choose"},{"chosen","choose"},{"chooses","choose"},{"choosing","choose"},
    {"lost","lose"},{"loses","lose"},{"losing","lose"},
    {"taught","teach"},{"teaches","teach"},{"teaching","teach"},
    {"spoke","speak"},{"spoken","speak"},{"speaks","speak"},{"speaking","speak"},
    {"caught","catch"},{"catches","catch"},{"catching","catch"},
    {"broke","break"},{"broken","break"},{"breaks","break"},{"breaking","break"},
    {"drew","draw"},{"drawn","draw"},{"draws","draw"},{"drawing","draw"},
    {"drove","drive"},{"driven","drive"},{"drives","drive"},{"driving","drive"},
    {"bought","buy"},{"buys","buy"},{"buying","buy"},
    {"ate","eat"},{"eaten","eat"},{"eats","eat"},{"eating","eat"},
    {"fell","fall"},{"fallen","fall"},{"falls","fall"},{"falling","fall"},
    {"flew","fly"},{"flown","fly"},{"flies","fly"},{"flying","fly"},
    {"raised","raise"},{"raises","raise"},{"raising","raise"},

    {"universities","university"},{"examinations","examination"},
    {"courses","course"},{"students","student"},{"sessions","session"},
    {"schedules","schedule"},{"registrations","registration"},
    {"applications","application"},{"opportunities","opportunity"},
    {"internships","internship"},{"placements","placement"},
    {"companies","company"},{"activities","activity"},
    {"communities","community"},{"categories","category"},
    {"policies","policy"},{"queries","query"},
    {"libraries","library"},{"facilities","facility"},
    {"offices","office"},{"updates","update"},
    {"meetings","meeting"},{"events","event"},
    {NULL, NULL}
};

static const char *lookup_irregular(const char *word) {
    for (int i = 0; irregulars[i].word; i++) {
        if (strcmp(word, irregulars[i].word) == 0)
            return irregulars[i].lemma;
    }
    return NULL;
}

static void suffix_lemmatize(const char *word, char *out, int cap) {
    int len = (int)strlen(word);
    strncpy(out, word, cap - 1);
    out[cap - 1] = '\0';

    if (len < 4) return;

    // -ies to -y
    if (len >= 4 && strcmp(word + len - 3, "ies") == 0) {
        strncpy(out, word, len - 3);
        out[len-3] = 'y';
        out[len-2] = '\0';
        return;
    }
    // -ied to -y
    if (len >= 4 && strcmp(word + len - 3, "ied") == 0) {
        strncpy(out, word, len - 3);
        out[len-3] = 'y';
        out[len-2] = '\0';
        return;
    }
    // -ing to base (handle doubled consonant)
    if (len >= 5 && strcmp(word + len - 3, "ing") == 0) {
        int base = len - 3;
        if (base >= 3 && word[base-1] == word[base-2] && !strchr("aeiou", word[base-1]))
            base--;
        strncpy(out, word, base);
        out[base] = '\0';
        return;
    }
    // -ed to base
    if (len >= 4 && strcmp(word + len - 2, "ed") == 0) {
        int base = len - 2;
        if (base >= 3 && word[base-1] == word[base-2] && !strchr("aeiou", word[base-1]))
            base--;
        strncpy(out, word, base);
        out[base] = '\0';
        return;
    }
    // trailing -s
    if (word[len-1] == 's' && word[len-2] != 's' && word[len-2] != 'u') {
        strncpy(out, word, len - 1);
        out[len-1] = '\0';
        return;
    }
}

static void lemmatize_word(const char *word, char *out, int cap) {
    static int firstCall = 1;
    if (firstCall) {
        printf(" [lemmatize] engaged\n");
        firstCall = 0;
    }

    const char *hit = lookup_irregular(word);
    if (hit) {
        strncpy(out, hit, cap - 1);
        out[cap - 1] = '\0';
        return;
    }
    suffix_lemmatize(word, out, cap);
}

#endif