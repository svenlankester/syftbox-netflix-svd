import os
from pathlib import Path

from syft_core import Client as SyftboxClient
from syft_core.permissions import SyftPermission

APP_NAME = os.getenv("APP_NAME", "syftbox-netflix-svd")


def setup_environment(client, app_name, aggregator_path, profile):
    """
    Set up public and private folders for data storage.

    Args:
        client: Client instance for managing APP and datasite paths.

    Returns:
        tuple: Paths to restricted public and private folders.
    """

    def create_private_folder(client: SyftboxClient, profile) -> Path:
        """
        Create a private folder within the specified path.

        This function creates a directory structure containing the NetflixViewingHistory.csv.
        """

        # private folders should always be out of the datasite path so that there are
        # no accidental data leaks
        APP_NAME = os.getenv("APP_NAME", "syftbox-netflix-svd")
        app_data_dir = Path(client.config.data_dir) / "private" / APP_NAME
        app_data_dir.mkdir(parents=True, exist_ok=True)
        netflix_datapath = app_data_dir / profile
        netflix_datapath.mkdir(parents=True, exist_ok=True)
        return netflix_datapath

    def create_public_folder(
        path: Path, client: SyftboxClient, aggregator_path
    ) -> None:
        """
        Create a API public folder within the specified path.

        This function creates a directory for receiving the private enhanced version \
        of the viewing history.
        """

        os.makedirs(path, exist_ok=True)

        # Set default permissions for this folder
        permissions = SyftPermission.datasite_default(context=client, dir=path)
        permissions.add_rule(
            path="**",
            user=aggregator_path,
            permission="read",
        )
        permissions.save(path)

    datasites_path = Path(client.datasite_path.parent)
    
    restricted_shared_folder = Path(
        datasites_path / aggregator_path / "app_data" / app_name / "shared"     # AGGREGADTOR's shared folder
    )
    
    restricted_public_folder = client.app_data(app_name) / profile

    create_public_folder(restricted_public_folder, client, aggregator_path)
    
    private_folder = create_private_folder(client, profile)

    return restricted_shared_folder, restricted_public_folder, private_folder
