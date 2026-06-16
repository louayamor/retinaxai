'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  ChevronDown, 
  ChevronUp, 
  Calendar, 
  Phone, 
  MapPin,
  User,
  AlertCircle,
  FileText,
  Scan,
  Eye
} from 'lucide-react';
import type { Patient } from '@/types';
import { slideInUp } from '@/lib/animations';

interface PatientCardProps {
  patient: Patient;
  onEdit?: (patient: Patient) => void;
  onDelete?: (id: string) => void;
}

export function PatientCard({ patient, onEdit, onDelete }: PatientCardProps) {
  const [expanded, setExpanded] = useState(false);
  const router = useRouter();

  const getInitials = (firstName: string, lastName: string) => {
    return `${firstName[0]}${lastName[0]}`.toUpperCase();
  };

  const getGenderClass = (gender: string) => {
    return gender === 'M' ? 'ring-blue-400' : 'ring-rose-400';
  };

  return (
    <motion.div variants={slideInUp}>
      <Card className="overflow-hidden hover:shadow-lg transition-all duration-300">
        {/* Card Header */}
        <CardHeader className="pb-3 bg-gradient-to-r from-muted/30 to-transparent">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              {/* Patient Avatar */}
              <div className={`flex h-12 w-12 items-center justify-center rounded-full bg-muted text-muted-foreground font-bold ring-2 ${getGenderClass(patient.gender)}`}>
                {getInitials(patient.first_name, patient.last_name)}
              </div>
              <div>
                <CardTitle className="text-lg">
                  {patient.first_name} {patient.last_name}
                </CardTitle>
                <div className="flex items-center gap-2 mt-1">
                  <Badge variant="outline" className="text-xs">
                    {patient.gender === 'M' ? 'Male' : 'Female'}
                  </Badge>
                  <span className="text-sm text-muted-foreground">
                    {patient.age} years old
                  </span>
                </div>
              </div>
            </div>
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={() => setExpanded(!expanded)}
              className="text-[var(--brand-teal)] hover:text-[#1a9a9a] hover:bg-[var(--brand-teal)]/10"
            >
              {expanded ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>
          </div>
        </CardHeader>

        {/* Card Content - Quick Info */}
        <CardContent className="pb-3">
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="bg-muted/50 rounded-lg p-2">
              <p className="text-xs text-muted-foreground">Medical Record #</p>
              <p className="font-mono font-medium">{patient.medical_record_number}</p>
            </div>
            <div className="bg-muted/50 rounded-lg p-2">
              <p className="text-xs text-muted-foreground">Registered</p>
              <p className="font-medium flex items-center gap-1">
                <Calendar className="h-3 w-3" />
                {new Date(patient.created_at).toLocaleDateString()}
              </p>
            </div>
          </div>
          
          <div className="flex items-center justify-between mt-3">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              {patient.ocr_patient_id && (
                <Badge variant="secondary" className="text-xs">
                  OCR: {patient.ocr_patient_id}
                </Badge>
              )}
            </div>
            <div className="flex gap-2">
              <Button 
                variant="outline" 
                size="sm" 
                onClick={() => router.push(`/dashboard/patients/${patient.id}`)}
              >
                <Eye className="h-4 w-4 mr-1" />
                View
              </Button>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={() => onEdit?.(patient)}
              >
                Edit
              </Button>
              <Button 
                variant="destructive" 
                size="sm" 
                onClick={() => onDelete?.(patient.id)}
              >
                Delete
              </Button>
            </div>
          </div>
        </CardContent>

        {/* Expanded Patient Detail */}
        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="border-t bg-muted/20"
            >
              <CardContent className="pt-4">
                <div className="space-y-4">
                  {/* Contact Information */}
                  <div className="bg-white dark:bg-slate-900 rounded-lg p-4">
                    <h4 className="font-semibold text-sm mb-3 flex items-center gap-2">
                      <User className="h-4 w-4 text-[var(--brand-teal)]" />
                      Contact Information
                    </h4>
                    <div className="grid gap-2 text-sm">
                      {patient.phone && (
                        <div className="flex items-center gap-2">
                          <Phone className="h-4 w-4 text-muted-foreground" />
                          <span>{patient.phone}</span>
                        </div>
                      )}
                      {patient.address && (
                        <div className="flex items-center gap-2">
                          <MapPin className="h-4 w-4 text-muted-foreground" />
                          <span>{patient.address}</span>
                        </div>
                      )}
                      {!patient.phone && !patient.address && (
                        <p className="text-muted-foreground text-sm">No contact information available</p>
                      )}
                    </div>
                  </div>

                  {/* Medical Record Details */}
                  <div className="bg-white dark:bg-slate-900 rounded-lg p-4">
                    <h4 className="font-semibold text-sm mb-3 flex items-center gap-2">
                      <FileText className="h-4 w-4 text-[var(--brand-teal)]" />
                      Medical Record Details
                    </h4>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <p className="text-xs text-muted-foreground">Medical Record Number</p>
                        <p className="font-mono">{patient.medical_record_number}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">OCR Patient ID</p>
                        <p className="font-mono">{patient.ocr_patient_id || '—'}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Date of Birth / Age</p>
                        <p>{patient.age} years old</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Gender</p>
                        <p>{patient.gender === 'M' ? 'Male' : 'Female'}</p>
                      </div>
                    </div>
                  </div>

                  {/* Quick Actions */}
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm">
                      <Scan className="h-4 w-4 mr-1" />
                      Add Scan
                    </Button>
                    <Button variant="outline" size="sm">
                      <FileText className="h-4 w-4 mr-1" />
                      View Predictions
                    </Button>
                  </div>
                </div>
              </CardContent>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>
    </motion.div>
  );
}