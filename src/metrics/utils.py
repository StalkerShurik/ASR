import fastwer

# Based on seminar materials

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    # print(target_text.shape, predicted_text.shape)
    return fastwer.score_sent(predicted_text, target_text, char_level=True)


def calc_wer(target_text, predicted_text) -> float:
    # print(target_text.shape, predicted_text.shape)
    return fastwer.score_sent(predicted_text, target_text)
