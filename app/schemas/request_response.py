# app/schemas/request_response.py

from datetime import date
from typing import Optional, Dict, Any, List

from pydantic import (
    BaseModel,
    Field,
    field_validator
)


class Location(BaseModel):

    latitude: float = Field(
        ...,
        ge=-90,
        le=90
    )

    longitude: float = Field(
        ...,
        ge=-180,
        le=180
    )


class ChangeDetectionRequest(BaseModel):

    latitude: float = Field(
        ...,
        ge=-90,
        le=90
    )

    longitude: float = Field(
        ...,
        ge=-180,
        le=180
    )

    before_date: date

    after_date: date

    max_cloud_cover: float = Field(
        default=20.0,
        ge=0,
        le=100
    )

    @field_validator(
        "after_date"
    )
    @classmethod
    def validate_dates(
        cls,
        value,
        info
    ):

        before_date = info.data.get(
            "before_date"
        )

        if (
            before_date is not None
            and value <= before_date
        ):

            raise ValueError(
                "after_date must be later "
                "than before_date"
            )

        return value


class ImageDates(BaseModel):

    requested_before: str

    requested_after: str

    actual_before: Optional[str] = None

    actual_after: Optional[str] = None


class ChangeDetectionResponse(BaseModel):

    status: str

    location: Location

    dates: ImageDates

    change_detection: Dict[str, Any]

    spatial_analysis: Dict[str, Any]

    explanation: str