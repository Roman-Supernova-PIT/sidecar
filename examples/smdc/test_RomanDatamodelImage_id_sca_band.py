"""Test that we can read band, observation_id, sca from a particular RomandDatamodelImage

You can check this by using less/more/cat/grep to just read the plain text of the ASDF header.
"""

from pathlib import Path

from snappl.image import RomanDatamodelImage

base_path = Path("/mnt/roman-science-east-2/snpit/snana+romanisim+romancal/output_images_SCAx2_ZYJHF_40day/")
science_image = Path("SNPIT_VISIT602000033_WFI01_F106_L2.asdf")

science_image_path = base_path / science_image

science_image = RomanDatamodelImage(full_filepath = science_image_path, no_base_path=True)

print(science_image.band)
print(science_image.observation_id)
print(science_image.sca)
