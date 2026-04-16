'use client';

import { useState, type ChangeEvent } from 'react';
import { Loader2 } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { toast } from 'sonner';
import { updatePatient } from '@/lib/api';
import type { Patient } from '@/types';

interface EditPatientDialogProps {
  patient: Patient | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

type PatientFormState = {
  first_name: string;
  last_name: string;
  age: string;
  gender: 'M' | 'F';
  medical_record_number: string;
  phone: string;
  address: string;
  ocr_patient_id: string;
};

const emptyForm: PatientFormState = {
  first_name: '',
  last_name: '',
  age: '',
  gender: 'M',
  medical_record_number: '',
  phone: '',
  address: '',
  ocr_patient_id: '',
};

export function EditPatientDialog({
  patient,
  open,
  onOpenChange,
  onSuccess,
}: EditPatientDialogProps) {
  const [form, setForm] = useState<PatientFormState>(emptyForm);
  const [saving, setSaving] = useState(false);

  const onChange =
    (key: keyof PatientFormState) =>
    (e: ChangeEvent<HTMLInputElement>) => {
      setForm((prev) => ({ ...prev, [key]: e.target.value }));
    };

  const handleOpenChange = (newOpen: boolean) => {
    if (newOpen && patient) {
      setForm({
        first_name: patient.first_name,
        last_name: patient.last_name,
        age: String(patient.age),
        gender: patient.gender,
        medical_record_number: patient.medical_record_number,
        phone: patient.phone ?? '',
        address: patient.address ?? '',
        ocr_patient_id: patient.ocr_patient_id ?? '',
      });
    }
    onOpenChange(newOpen);
  };

  const onSubmit = async () => {
    if (!patient) return;

    if (!form.first_name || !form.last_name || !form.medical_record_number || !form.age) {
      toast.error('Please fill in all required fields');
      return;
    }

    const ageNum = Number(form.age);
    if (isNaN(ageNum) || ageNum < 0 || ageNum > 150) {
      toast.error('Please enter a valid age between 0 and 150');
      return;
    }

    setSaving(true);
    try {
      await updatePatient(patient.id, {
        first_name: form.first_name,
        last_name: form.last_name,
        age: ageNum,
        gender: form.gender,
        medical_record_number: form.medical_record_number,
        phone: form.phone || null,
        address: form.address || null,
        ocr_patient_id: form.ocr_patient_id || null,
      });

      toast.success('Patient updated successfully', {
        icon: (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5 text-emerald-500"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
            <polyline points="22 4 12 14.01 9 11.01" />
          </svg>
        ),
        className: 'border-l-4 border-l-emerald-500',
      });

      onOpenChange(false);
      onSuccess();
    } catch (err) {
      console.error('Failed to update patient:', err);
      toast.error('Failed to update patient', {
        description: err instanceof Error ? err.message : 'Please try again',
        icon: (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5 text-red-500"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="15" x2="9" y1="9" y2="15" />
            <line x1="9" x2="15" y1="9" y2="15" />
          </svg>
        ),
      });
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5 text-[var(--brand-teal)]"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
              <path d="m15 5 4 4" />
            </svg>
            Edit Patient
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <label className="text-sm font-medium mb-1 block">
                First Name <span className="text-destructive">*</span>
              </label>
              <Input
                placeholder="Enter first name"
                value={form.first_name}
                onChange={onChange('first_name')}
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-1 block">
                Last Name <span className="text-destructive">*</span>
              </label>
              <Input
                placeholder="Enter last name"
                value={form.last_name}
                onChange={onChange('last_name')}
              />
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <label className="text-sm font-medium mb-1 block">
                Age <span className="text-destructive">*</span>
              </label>
              <Input
                placeholder="Age"
                type="number"
                min={0}
                value={form.age}
                onChange={onChange('age')}
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-1 block">Gender</label>
              <div className="flex gap-2">
                <Button
                  type="button"
                  variant={form.gender === 'M' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setForm({ ...form, gender: 'M' })}
                  className={`flex-1 ${form.gender === 'M' ? 'bg-blue-500 hover:bg-blue-600' : ''}`}
                >
                  Male
                </Button>
                <Button
                  type="button"
                  variant={form.gender === 'F' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setForm({ ...form, gender: 'F' })}
                  className={`flex-1 ${form.gender === 'F' ? 'bg-pink-500 hover:bg-pink-600' : ''}`}
                >
                  Female
                </Button>
              </div>
            </div>
          </div>

          <div>
            <label className="text-sm font-medium mb-1 block">
              Medical Record Number <span className="text-destructive">*</span>
            </label>
            <Input
              placeholder="e.g., MRN-2024-001"
              value={form.medical_record_number}
              onChange={onChange('medical_record_number')}
            />
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <label className="text-sm font-medium mb-1 block">Phone</label>
              <Input
                placeholder="+1 234 567 8900"
                value={form.phone}
                onChange={onChange('phone')}
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-1 block">OCR Patient ID</label>
              <Input
                placeholder="From OCT report"
                value={form.ocr_patient_id}
                onChange={onChange('ocr_patient_id')}
              />
            </div>
          </div>

          <div>
            <label className="text-sm font-medium mb-1 block">Address</label>
            <Input
              placeholder="Full address"
              value={form.address}
              onChange={onChange('address')}
            />
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={saving}>
            Cancel
          </Button>
          <Button onClick={onSubmit} disabled={saving}>
            {saving ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="mr-2 h-4 w-4"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <polyline points="20 6 9 17 4 12" />
                </svg>
                Save Changes
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
