import click


@click.command()
def transform_core():
    from dataset import dataset_generator
    dataset_generator.main()


@click.group()
def main():
    pass


main.add_command(transform_core)

if __name__ == '__main__':
    main()
