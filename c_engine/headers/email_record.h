#ifndef EMAIL_RECORD_H
#define EMAIL_RECORD_H

#define MAX_SENDER_EMAIL   256
#define MAX_RECEIVER_EMAIL 512
#define MAX_TIMESTAMP      64
#define MAX_SUBJECT        512
#define MAX_BODY           65536

typedef struct {
    char senderEmail   [MAX_SENDER_EMAIL];
    char receiverEmail [MAX_RECEIVER_EMAIL];
    char timestamp     [MAX_TIMESTAMP];
    char subject       [MAX_SUBJECT];
    char body          [MAX_BODY];
} EmailRecord;

#endif