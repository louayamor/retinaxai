'use client';

import { useEffect, useState, type ChangeEvent } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { createPatient, getPatients, getPatientStats } from '@/lib/api';
import PageContainer from '@/components/layout/page-container';
import { PageHero } from '@/components/ui/page-hero';
import { PageSection } from '@/components/ui/page-section';
import { StatsRow } from '@/components/ui/stats-row';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { toast } from 'sonner';
import type { Patient } from '@/types';
import { fadeInUp, slideInUp, buttonTap } from '@/lib/animations';
import { StatsCard } from '@/components/ui/stats-card';
import { PatientTable } from '@/components/patients/PatientTable';
import { EditPatientDialog } from '@/components/patients/EditPatientDialog';
import { DeleteConfirmModal } from '@/components/patients/DeleteConfirmModal';
import { PatientFilters } from '@/components/patients/PatientFilters';
import { Users, UserPlus, Calendar, UserCheck, Search, X, User, Loader2 } from 'lucide-react';

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

const generateMRN = () => {
  const year = new Date().getFullYear();
  return `MRN-${year}-`;
};

const generateOCRId = (existingPatients: Patient[]) => {
  const existingIds = existingPatients
    .map(p => p.ocr_patient_id)
    .filter(id => id && !isNaN(Number(id)))
    .map(Number);
  const maxId = existingIds.length > 0 ? Math.max(...existingIds) : 0;
  return String(maxId + 1);
};

const createEmptyForm = (patients: Patient[]): PatientFormState => ({
  first_name: '',
  last_name: '',
  age: '',
  gender: 'M',
  medical_record_number: generateMRN(),
  phone: '',
  address: '',
  ocr_patient_id: generateOCRId(patients)
});

export default function PatientsPage() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [stats, setStats] = useState<{
    total: number;
    avg_age: number;
    male_count: number;
    female_count: number;
    this_month: number;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [search, setSearch] = useState('');
  const [form, setForm] = useState<PatientFormState>(() => createEmptyForm([]));
  const [genderFilter, setGenderFilter] = useState<'all' | 'M' | 'F'>('all');
  const [editPatient, setEditPatient] = useState<Patient | null>(null);
  const [deletePatient, setDeletePatient] = useState<Patient | null>(null);
  const shouldReduceMotion = useReducedMotion();

  const loadStats = async () => {
    try {
      const data = await getPatientStats();
      setStats(data);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    }
  };

  const loadPatients = async () => {
    setLoading(true);
    try {
      const data = await getPatients();
      const patientsArray = Array.isArray(data) ? data : [];
      setPatients(patientsArray);
      if (form.medical_record_number === generateMRN()) {
        setForm(createEmptyForm(patientsArray));
      }
    } catch (err) {
      console.error('Failed to fetch patients:', err);
      setPatients([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadPatients();
    void loadStats();
  }, []);

  useEffect(() => {
    void loadPatients();
  }, [search]);

  const onChange =
    (key: keyof PatientFormState) =>
    (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
      setForm((prev) => ({ ...prev, [key]: e.target.value }));
    };

  const resetForm = () => {
    setForm(createEmptyForm(patients));
  };

  const onSubmit = async () => {
    if (!form.first_name || !form.last_name || !form.medical_record_number || !form.age) {
      toast.error('Please fill in all required fields: First name, Last name, Age, and Medical Record Number');
      return;
    }

    const ageNum = Number(form.age);
    if (isNaN(ageNum) || ageNum < 0 || ageNum > 150) {
      toast.error('Please enter a valid age between 0 and 150');
      return;
    }

    setSaving(true);
    try {
      const created = await createPatient({
        first_name: form.first_name,
        last_name: form.last_name,
        age: ageNum,
        gender: form.gender,
        medical_record_number: form.medical_record_number,
        phone: form.phone || null,
        address: form.address || null,
        ocr_patient_id: form.ocr_patient_id || null
      });
      toast.success('Patient created successfully');
      setPatients(prev => [...prev, { ...created, created_at: new Date().toISOString() }]);
      await loadStats();
      resetForm();
    } catch (err) {
      console.error('Failed to save patient:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to save patient';
      toast.error(errorMessage);
    } finally {
      setSaving(false);
    }
  };

  const handleEdit = (patient: Patient) => {
    setEditPatient(patient);
  };

  const handleDelete = (patient: Patient) => {
    setDeletePatient(patient);
  };

  const handleEditSuccess = () => {
    void loadPatients();
  };

  const handleDeleteSuccess = () => {
    void loadPatients();
    void loadStats();
  };

  return (
    <PageContainer className='flex flex-col gap-6'>
      <motion.div
        variants={shouldReduceMotion ? {} : fadeInUp}
        initial='hidden'
        animate='visible'
        className='flex flex-col gap-6'
      >
        <PageHero
          title='Patient Registry'
          description='Manage patient records, OCT scans, and clinical data'
          imageUrl='https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=800&q=80'
        />

        <StatsRow columns={4}>
          <StatsCard
            title='Total Patients'
            value={stats?.total ?? 0}
            icon={Users}
            subtitle='In database'
          />
          <StatsCard
            title='New This Month'
            value={stats?.this_month ?? 0}
            icon={UserPlus}
            color='#22c55e'
          />
          <StatsCard
            title='Avg Age'
            value={stats?.avg_age ?? 0}
            icon={Calendar}
            color='#3b82f6'
            subtitle='years'
          />
          <StatsCard
            title='Gender Split'
            value={`${stats?.male_count ?? 0} M / ${stats?.female_count ?? 0} F`}
            icon={UserCheck}
            color='var(--brand-gold)'
          />
        </StatsRow>

        <PageSection
          title='Add New Patient'
          icon={<UserPlus className='h-5 w-5' />}
        >
          <CardContent className='space-y-4'>
            <div className='grid gap-3 md:grid-cols-2 lg:grid-cols-4'>
              <div>
                <label className='text-sm font-medium mb-1 block'>First Name *</label>
                <Input
                  placeholder='Enter first name'
                  value={form.first_name}
                  onChange={onChange('first_name')}
                />
              </div>
              <div>
                <label className='text-sm font-medium mb-1 block'>Last Name *</label>
                <Input
                  placeholder='Enter last name'
                  value={form.last_name}
                  onChange={onChange('last_name')}
                />
              </div>
              <div>
                <label className='text-sm font-medium mb-1 block'>Age *</label>
                <Input
                  placeholder='Age'
                  type='number'
                  min={0}
                  value={form.age}
                  onChange={onChange('age')}
                />
              </div>
              <div>
                <label className='text-sm font-medium mb-1 block'>Gender</label>
                <div className='flex gap-2'>
                  <Button
                    type='button'
                    variant={form.gender === 'M' ? 'default' : 'outline'}
                    size='sm'
                    onClick={() => setForm({ ...form, gender: 'M' })}
                    className={`flex-1 ${form.gender === 'M' ? 'bg-blue-500 hover:bg-blue-600' : ''}`}
                  >
                    <User className='mr-1 h-4 w-4' /> Male
                  </Button>
                  <Button
                    type='button'
                    variant={form.gender === 'F' ? 'default' : 'outline'}
                    size='sm'
                    onClick={() => setForm({ ...form, gender: 'F' })}
                    className={`flex-1 ${form.gender === 'F' ? 'bg-pink-500 hover:bg-pink-600' : ''}`}
                  >
                    <User className='mr-1 h-4 w-4' /> Female
                  </Button>
                </div>
              </div>
            </div>

            <div className='grid gap-3 md:grid-cols-2 lg:grid-cols-4'>
              <div className='lg:col-span-2'>
                <label className='text-sm font-medium mb-1 block'>Medical Record Number *</label>
                <Input
                  placeholder='e.g., MRN-2024-001'
                  value={form.medical_record_number}
                  onChange={onChange('medical_record_number')}
                />
              </div>
              <div>
                <label className='text-sm font-medium mb-1 block'>OCR Patient ID</label>
                <Input
                  placeholder='From OCT report'
                  value={form.ocr_patient_id}
                  onChange={onChange('ocr_patient_id')}
                />
              </div>
              <div>
                <label className='text-sm font-medium mb-1 block'>Phone</label>
                <Input
                  placeholder='+1 234 567 8900'
                  value={form.phone}
                  onChange={onChange('phone')}
                />
              </div>
            </div>

            <div>
              <label className='text-sm font-medium mb-1 block'>Address</label>
              <Input
                placeholder='Full address'
                value={form.address}
                onChange={onChange('address')}
              />
            </div>

            <div className='flex gap-2 pt-2'>
              <motion.div variants={shouldReduceMotion ? {} : buttonTap}>
                <Button onClick={onSubmit} disabled={saving}>
                  {saving ? (
                    <>
                      <Loader2 className='mr-2 h-4 w-4 animate-spin' />
                      Adding...
                    </>
                  ) : (
                    <>
                      <span className='mr-2'>+</span>
                      Add Patient
                    </>
                  )}
                </Button>
              </motion.div>
            </div>
          </CardContent>
        </PageSection>

        <PageSection
          title='Patient List'
          description={`${patients.length} patients`}
          padding='none'
          headerAction={
            <Button
              variant='outline'
              size='sm'
              onClick={() => {
                document.getElementById('add-patient-form')?.scrollIntoView({ behavior: 'smooth' });
              }}
            >
              <UserPlus className='mr-2 h-4 w-4' />
              Add Patient
            </Button>
          }
        >
          <div className='space-y-4 p-4 md:p-6'>
            <div className='flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4'>
              <PatientFilters
                total={stats?.total ?? 0}
                maleCount={stats?.male_count ?? 0}
                femaleCount={stats?.female_count ?? 0}
                value={genderFilter}
                onChange={setGenderFilter}
              />
              <div className='relative flex-1 sm:flex-initial sm:w-72'>
                <Search className='absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground' />
                <Input
                  placeholder='Search by name, MRN...'
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className='pl-9'
                />
                {search && (
                  <button
                    onClick={() => setSearch('')}
                    className='absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground'
                  >
                    <X className='h-4 w-4' />
                  </button>
                )}
              </div>
            </div>

            <PatientTable
              patients={patients}
              loading={loading}
              onEdit={handleEdit}
              onDelete={handleDelete}
              genderFilter={genderFilter}
              searchQuery={search}
            />
          </div>
        </PageSection>
      </motion.div>

      <EditPatientDialog
        patient={editPatient}
        open={!!editPatient}
        onOpenChange={(open) => !open && setEditPatient(null)}
        onSuccess={handleEditSuccess}
      />

      <DeleteConfirmModal
        patient={deletePatient}
        open={!!deletePatient}
        onOpenChange={(open) => !open && setDeletePatient(null)}
        onSuccess={handleDeleteSuccess}
      />
    </PageContainer>
  );
}
