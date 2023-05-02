from mrisd import sequences
import logging


def main():
    seq = sequences.mese()
    seq.plot(add_magnetization=False, save=f'mese_seq_diagram.png')


if __name__ == '__main__':
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    try:
        main()
    except (AttributeError, ValueError) as e:
        logging.error(e)
