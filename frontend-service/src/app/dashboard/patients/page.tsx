'use client';

import { useEffect, useState, type ChangeEvent } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { createPatient, getPatients, getPatientStats } from '@/lib/api';
import PageContainer from '@/components/layout/page-container';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { toast } from 'sonner';
import Image from 'next/image';
import type { Patient } from '@/types';
import { fadeInUp, slideInUp, staggerContainer, staggerItemFast, buttonTap, rowHover } from '@/lib/animations';
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

const emptyForm: PatientFormState = {
  first_name: '',
  last_name: '',
  age: '',
  gender: 'M',
  medical_record_number: '',
  phone: '',
  address: '',
  ocr_patient_id: ''
};

export default function PatientsPage() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [stats, setStats] = useState<{ total: number; avg_age: number; male_count: number; female_count: number; this_month: number } | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [search, setSearch] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState<PatientFormState>(emptyForm);
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
      setPatients(Array.isArray(data) ? data : []);
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
    setForm(emptyForm);
    setEditingId(null);
  };

  const onSubmit = async () => {
    // Validation
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
      toast.success('Patient created successfully', {
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
    <PageContainer>
      <motion.div
        variants={shouldReduceMotion ? {} : fadeInUp}
        initial="hidden"
        animate="visible"
        className="flex flex-col gap-8"
      >
        {/* Hero with Medical Image */}
        <motion.div
          variants={shouldReduceMotion ? {} : slideInUp}
          className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-[#0a2e3e] via-[#0d3a4c] to-[#104a5e] p-10 text-white"
        >
          <div className="absolute right-0 top-0 h-full w-1/3 opacity-20">
            <Image
              src="https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=800&q=80"
              alt="Patient Care"
              fill
              className="object-cover"
              unoptimized
            />
          </div>
          {/* Decorative accent shapes */}
          <div className="absolute -bottom-20 -left-20 h-40 w-40 rounded-full bg-[var(--brand-teal)]/10 blur-3xl" />
          <div className="absolute top-10 right-20 h-24 w-24 rounded-full bg-[var(--brand-gold)]/10 blur-2xl" />
          
          <div className="relative z-10">
            <h1 className="mb-2 text-3xl font-bold tracking-tight">Patient Registry</h1>
            <p className="max-w-xl text-lg text-white/70">
              Manage patient records, OCT scans, and clinical data
            </p>
          </div>
        </motion.div>

        {/* Stats Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatsCard
            title="Total Patients"
            value={stats?.total ?? 0}
            icon={Users}
            subtitle="In database"
          />
          <StatsCard
            title="New This Month"
            value={stats?.this_month ?? 0}
            icon={UserPlus}
            color="#22c55e"
          />
          <StatsCard
            title="Avg Age"
            value={stats?.avg_age ?? 0}
            icon={Calendar}
            color="#3b82f6"
            subtitle="years"
          />
          <StatsCard
            title="Gender Split"
            value={`${stats?.male_count ?? 0} M / ${stats?.female_count ?? 0} F`}
            icon={UserCheck}
            color="var(--brand-gold)"
          />
        </div>

        {/* Patient Form */}
        <motion.div variants={shouldReduceMotion ? {} : slideInUp} id="add-patient-form">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <UserPlus className="h-5 w-5 text-[var(--brand-teal)]" />
                Add New Patient
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Personal Info */}
              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
                <div>
                  <label className="text-sm font-medium mb-1 block">First Name *</label>
                  <Input 
                    placeholder="Enter first name" 
                    value={form.first_name} 
                    onChange={onChange('first_name')} 
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-1 block">Last Name *</label>
                  <Input 
                    placeholder="Enter last name" 
                    value={form.last_name} 
                    onChange={onChange('last_name')} 
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-1 block">Age *</label>
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
                      <User className="mr-1 h-4 w-4" /> Male
                    </Button>
                    <Button
                      type="button"
                      variant={form.gender === 'F' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setForm({ ...form, gender: 'F' })}
                      className={`flex-1 ${form.gender === 'F' ? 'bg-pink-500 hover:bg-pink-600' : ''}`}
                    >
                      <User className="mr-1 h-4 w-4" /> Female
                    </Button>
                  </div>
                </div>
              </div>
              
              {/* Medical Record Info */}
              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
                <div className="lg:col-span-2">
                  <label className="text-sm font-medium mb-1 block">Medical Record Number *</label>
                  <Input 
                    placeholder="e.g., MRN-2024-001" 
                    value={form.medical_record_number}
                    onChange={onChange('medical_record_number')}
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
                <div>
                  <label className="text-sm font-medium mb-1 block">Phone</label>
                  <Input 
                    placeholder="+1 234 567 8900" 
                    value={form.phone} 
                    onChange={onChange('phone')} 
                  />
                </div>
              </div>

              {/* Address */}
              <div>
                <label className="text-sm font-medium mb-1 block">Address</label>
                <Input 
                  placeholder="Full address" 
                  value={form.address} 
                  onChange={onChange('address')} 
                />
              </div>

              {/* Actions */}
              <div className="flex gap-2 pt-2">
                <motion.div variants={shouldReduceMotion ? {} : buttonTap}>
                  <Button onClick={onSubmit} disabled={saving}>
                    {saving ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Adding...
                      </>
                    ) : (
                      <>
                        <span className="mr-2">+</span>
                        Add Patient
                      </>
                    )}
                  </Button>
                </motion.div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Search & Patient List */}
        <motion.div variants={shouldReduceMotion ? {} : fadeInUp} className="space-y-4">
          {/* Search & Filters */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <PatientFilters
              total={stats?.total ?? 0}
              maleCount={stats?.male_count ?? 0}
              femaleCount={stats?.female_count ?? 0}
              value={genderFilter}
              onChange={setGenderFilter}
            />
            <div className="flex items-center gap-4 w-full sm:w-auto">
              <div className="relative flex-1 sm:flex-initial sm:w-72">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search by name, MRN..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="pl-9"
                />
                {search && (
                  <button
                    onClick={() => setSearch('')}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    <X className="h-4 w-4" />
                  </button>
                )}
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  const form = document.getElementById('add-patient-form');
                  form?.scrollIntoView({ behavior: 'smooth' });
                }}
              >
                <UserPlus className="mr-2 h-4 w-4" />
                Add Patient
              </Button>
            </div>
          </div>

          {/* Patient Table */}
          <PatientTable
            patients={patients}
            loading={loading}
            onEdit={handleEdit}
            onDelete={handleDelete}
            genderFilter={genderFilter}
            searchQuery={search}
          />
        </motion.div>

        {/* Edit Dialog */}
        <EditPatientDialog
          patient={editPatient}
          open={!!editPatient}
          onOpenChange={(open) => !open && setEditPatient(null)}
          onSuccess={handleEditSuccess}
        />

        {/* Delete Confirmation */}
        <DeleteConfirmModal
          patient={deletePatient}
          open={!!deletePatient}
          onOpenChange={(open) => !open && setDeletePatient(null)}
          onSuccess={handleDeleteSuccess}
        />
      </motion.div>
    </PageContainer>
  );
}
