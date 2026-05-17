from pydantic import BaseModel, Field
from typing import Optional


class ReportMetadata(BaseModel):
    device: Optional[str] = None
    report_type: Optional[str] = None
    eye: Optional[str] = None
    capture_date: Optional[str] = None
    print_date: Optional[str] = None
    image_quality: Optional[int] = None
    analysis_mode: Optional[str] = None
    scan_protocol: Optional[str] = None
    fixation: Optional[str] = None


class PatientInfo(BaseModel):
    patient_id: Optional[str] = None
    gender: Optional[str] = None
    dob: Optional[str] = None
    age: Optional[int] = None
    ethnicity: Optional[str] = None


class RetinalThickness(BaseModel):
    center_fovea: Optional[float] = None
    inner_superior: Optional[float] = None
    inner_nasal: Optional[float] = None
    inner_inferior: Optional[float] = None
    inner_temporal: Optional[float] = None
    outer_superior: Optional[float] = None
    outer_nasal: Optional[float] = None
    outer_inferior: Optional[float] = None
    outer_temporal: Optional[float] = None
    average_thickness: Optional[float] = None
    center_thickness: Optional[float] = None
    total_volume_mm3: Optional[float] = None


class ClinicalFindings(BaseModel):
    vitreous: Optional[str] = None
    vessels: Optional[str] = None
    thickness_note: Optional[str] = None
    central_retina: Optional[str] = None
    laser_marks: Optional[bool] = None
    edema: Optional[bool] = None
    npdr_grade: Optional[str] = None
    erm_status: Optional[str] = None


class RegionImage(BaseModel):
    png_path: str
    base64_png: str


class OCTReport(BaseModel):
    source_file: str
    metadata: ReportMetadata = Field(default_factory=ReportMetadata)
    patient: PatientInfo = Field(default_factory=PatientInfo)
    thickness: RetinalThickness = Field(default_factory=RetinalThickness)
    clinical: ClinicalFindings = Field(default_factory=ClinicalFindings)
    images: dict[str, RegionImage] = Field(default_factory=dict)
    raw_text: Optional[str] = None
    extraction_warnings: list[str] = Field(default_factory=list)
