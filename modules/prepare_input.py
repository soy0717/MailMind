import csv
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger('data preparation')

def prepare_data(csvFile, outFile):
    num_rows = 0
    with open(csvFile, encoding='utf-8') as fin, open(outFile, 'w', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        for i, row in enumerate(reader):
            emailID = f"EML_{i:05d}"
            domain = row.get('trueDomain', '').strip()
            subject = row.get('subject', '')
            body = row.get('content', '')

            # just concat and clean
            text = subject + ' ' + body
            text = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')

            fout.write(f"{emailID}\t{domain}\t{text}\n")
            num_rows += 1

    return num_rows

if __name__ == '__main__':
    rootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inPath = os.path.join(rootDir, 'data', 'emails_new.csv')
    outPath = os.path.join(rootDir, 'data', 'raw_emails.tsv')

    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    logger.info(f"Reading from {inPath}...")

    # process it
    numRows = prepare_data(inPath, outPath)
    logger.info(f"Success! Wrote {numRows} lines to {outPath}")
