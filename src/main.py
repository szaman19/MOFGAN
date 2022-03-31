import click


@click.command()
def generate_dataset():
    from dataset import dataset_generator
    dataset_generator.main()


@click.group()
def main():
    pass


main.add_command(generate_dataset)

if __name__ == '__main__':
    main()
