from testers.collecting_testers import test_cc_collector, test_hf_collector, test_s2ocr_collector, test_book_collector, test_wiki_collector
from testers.anonymzing_testers import test_anonymizer
from testers.quality_filtering_testers import test_quality_filter
from multiprocessing import set_start_method


def main():
    test_cc_collector()
    # test_hf_collector()
    # test_wiki_collector()
    # test_s2ocr_collector()
    # test_book_collector()
    # test_anonymizer()
    # test_quality_filter()


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except:
        print("context already set")
    main()