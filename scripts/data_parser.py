import argparse
import logging
import json
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    parse_file(args.train_file_name, "train_in.txt", "train_ner.txt")
    parse_file(args.test_file_name, "test_in.txt", "test_ner.txt")
    print ("DONE")
    return


def parse_file(input_filename, output_raw_filename, output_tag_filename):
    # Open file
    with open(args.data_dir + input_filename) as data_file:
        data = json.load(data_file)
    sentences = []
    target_output = []
    for entry in data:
        sentence = entry["ground_truth"]["query"]
        # Add to sentences arrray
        sentences.append(sentence)

        # Generate formated ourput
        output = ""
        sentences_words = sentence.split()
        i = 0
        while i < (len(sentences_words)):
            # check entities
            token_buffer = ""
            for entity in entry["ground_truth"]["entities"]:
                for entityWord in entity["literal"].split():
                    if i < (len(sentences_words)):
                        if sentences_words[i] == entityWord:
                            token_buffer += (entity["key"] + " ")
                            i += 1
                        else:
                            break
                    else:
                        break
            if token_buffer != "":
                output += token_buffer
            else:
                output += ("_PASS" + " ")
                i += 1
        target_output.append(output)

        # Save to file
        raw_file = open(args.output_dir + output_raw_filename, 'w')
        for item in sentences:
            raw_file.write("%s\n" % item)
        tag_file = open(args.output_dir + output_tag_filename, 'w')
        for item in target_output:
            tag_file.write("%s\n" % item)

        raw_file.close()
        tag_file.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for parsing the arguments")
    parser.add_argument('--data-dir', type=str, default="../data/", help='path to file')
    parser.add_argument('--output-dir', type=str, default="../data/format_data/", help="path to output")
    parser.add_argument('--train-file-name', type=str, default="train_data.json", help="training file name")
    parser.add_argument('--test-file-name', type=str, default="test_data.json", help="testing file name")
    args = parser.parse_args()
    logging.info(args)
    path = os.getcwd()+"/"+args.output_dir
    if not os.path.exists(path):
        os.makedirs(path)
    main()
