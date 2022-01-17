import hanzidentifier
from zhconv import convert

def convert_Chinese(sentence, nertags):
    output_list = []
    chinese_cha_sequence = []

    # Simplyfied chinese translation
    for one_mention in sentence:
        if (hanzidentifier.has_chinese(one_mention) or "·" == one_mention) and len(one_mention) == 1:
            chinese_cha_sequence.append(one_mention)
        else:
            if len(chinese_cha_sequence) > 0:
                chinese_cha_sequence = "".join(chinese_cha_sequence)
                chinese_cha_sequence = convert(chinese_cha_sequence, 'zh-cn')
                chinese_cha_sequence = list(chinese_cha_sequence)
                output_list.extend(chinese_cha_sequence)
            output_list.append(one_mention)
            chinese_cha_sequence = []

    if len(chinese_cha_sequence) > 0:
        chinese_cha_sequence = "".join(chinese_cha_sequence)
        chinese_cha_sequence = convert(chinese_cha_sequence, 'zh-cn')
        chinese_cha_sequence = list(chinese_cha_sequence)
        output_list.extend(chinese_cha_sequence)

    # Update the ner_tags for the translated chinese.
    if len(output_list) == len(nertags):
        return output_list, nertags
    if len(output_list) < len(nertags):
        return output_list, nertags[:len(nertags)]
    if len(output_list) > len(nertags):
        added_tokens_number = len(output_list) - len(nertags)
        if nertags[0] == 0:
            added_tokens = 0
        else:
            added_tokens = int(nertags[0]) + 1
        for i in range(added_tokens_number):
            nertags.append(added_tokens)
        return output_list, nertags


def chinese_translation(raw_datasets):
    all_tokens = []
    all_ner_tags = []
    for tokens,ner_tags in zip(raw_datasets["train"]['tokens'], raw_datasets["train"]['ner_tags']):

        sentence_segmentation_mention_list = []
        ner_tags_list = []

        current_sentence_segmentation_mention_list = []
        current_ner_tags_list = []

        short_memory = 100

        for token,ner_tag in zip(tokens,ner_tags):

            if int(short_memory) == int(ner_tag) and (int(ner_tag)%2 == 0):
                current_sentence_segmentation_mention_list.append(token)
                current_ner_tags_list.append(ner_tag)
                short_memory = ner_tag
            elif (int(short_memory) == (int(ner_tag) - 1)) and (int(ner_tag)%2 == 0):
                current_sentence_segmentation_mention_list.append(token)
                current_ner_tags_list.append(ner_tag)
                short_memory = ner_tag
            elif short_memory == 100:
                current_sentence_segmentation_mention_list.append(token)
                current_ner_tags_list.append(ner_tag)
                short_memory = ner_tag
            else:
                # Translation for a sequence, each sequence consists tokens with same tag.
                Chinese_translation = convert_Chinese(current_sentence_segmentation_mention_list, current_ner_tags_list)
                sentence_segmentation_mention_list.extend(Chinese_translation[0])
                ner_tags_list.extend(Chinese_translation[1])
                # Begining a new sequence
                current_sentence_segmentation_mention_list = [token]
                current_ner_tags_list = [ner_tag]
                short_memory = ner_tag

        Chinese_translation = convert_Chinese(current_sentence_segmentation_mention_list, current_ner_tags_list)
        sentence_segmentation_mention_list.extend(Chinese_translation[0])
        ner_tags_list.extend(Chinese_translation[1])

        all_tokens.append(sentence_segmentation_mention_list)
        all_ner_tags.append(ner_tags_list)
    return all_tokens, all_ner_tags

if __name__ == '__main__':
    output = ["不", "限", "於", "PC", "占", "士", "邦", "电", "影"]
    tages = [1, 2, 2, 2, 2, 2, 2, 2, 2]
    print(convert_Chinese(output, tages))
    