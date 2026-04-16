'use client';

import { useRouter } from 'next/navigation';
import type { ColumnDef } from '@tanstack/react-table';
import { ArrowUpDown, ChevronUp, ChevronDown } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Patient } from '@/types';

interface PatientCellProps {
  patient: Patient;
}

function PatientCell({ patient }: PatientCellProps) {
  const router = useRouter();

  const getInitials = (firstName: string, lastName: string) => {
    return `${firstName[0]}${lastName[0]}`.toUpperCase();
  };

  const getGenderColor = (gender: string) => {
    return gender === 'M'
      ? 'bg-gradient-to-br from-blue-500 to-blue-700'
      : 'bg-gradient-to-br from-pink-500 to-pink-700';
  };

  return (
    <button
      onClick={() => router.push(`/dashboard/patients/${patient.id}`)}
      className="flex items-center gap-3 text-left hover:opacity-80 transition-opacity"
    >
      <div
        className={`flex h-8 w-8 items-center justify-center rounded-full ${getGenderColor(
          patient.gender
        )} text-white text-xs font-bold`}
      >
        {getInitials(patient.first_name, patient.last_name)}
      </div>
      <span className="font-medium">
        {patient.first_name} {patient.last_name}
      </span>
    </button>
  );
}

interface SortableHeaderProps {
  column: { getIsSorted: () => false | 'asc' | 'desc'; toggleSorting: (desc?: boolean) => void };
  title: string;
}

function SortableHeader({ column, title }: SortableHeaderProps) {
  const sorted = column.getIsSorted();

  return (
    <Button
      variant="ghost"
      onClick={() => column.toggleSorting(sorted === 'asc')}
      className="-ml-3 h-8 px-2 hover:bg-transparent hover:text-foreground"
    >
      <span>{title}</span>
      {sorted === 'asc' ? (
        <ChevronUp className="ml-1 h-4 w-4" />
      ) : sorted === 'desc' ? (
        <ChevronDown className="ml-1 h-4 w-4" />
      ) : (
        <ArrowUpDown className="ml-1 h-4 w-4 opacity-50" />
      )}
    </Button>
  );
}

interface ActionCellProps {
  patient: Patient;
  onEdit: (patient: Patient) => void;
  onDelete: (patient: Patient) => void;
}

function ActionCell({ patient, onEdit, onDelete }: ActionCellProps) {
  return (
    <div className="flex items-center gap-1">
      <button
        onClick={(e) => {
          e.stopPropagation();
          onEdit(patient);
        }}
        className="flex h-8 w-8 items-center justify-center rounded-full hover:bg-muted transition-colors"
        title="Edit patient"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="text-muted-foreground hover:text-foreground"
        >
          <path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
          <path d="m15 5 4 4" />
        </svg>
      </button>
      <button
        onClick={(e) => {
          e.stopPropagation();
          onDelete(patient);
        }}
        className="flex h-8 w-8 items-center justify-center rounded-full hover:bg-destructive/10 transition-colors"
        title="Delete patient"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="text-muted-foreground hover:text-destructive"
        >
          <path d="M3 6h18" />
          <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
          <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
          <line x1="10" x2="10" y1="11" y2="17" />
          <line x1="14" x2="14" y1="11" y2="17" />
        </svg>
      </button>
    </div>
  );
}

export function createPatientColumns(
  onEdit: (patient: Patient) => void,
  onDelete: (patient: Patient) => void
): ColumnDef<Patient>[] {
  return [
    {
      accessorKey: 'patient',
      id: 'patient',
      header: ({ column }) => <SortableHeader column={column} title="Patient" />,
      cell: ({ row }) => <PatientCell patient={row.original} />,
      sortingFn: (rowA, rowB) => {
        const nameA = `${rowA.original.first_name} ${rowA.original.last_name}`.toLowerCase();
        const nameB = `${rowB.original.first_name} ${rowB.original.last_name}`.toLowerCase();
        return nameA.localeCompare(nameB);
      },
    },
    {
      accessorKey: 'gender',
      header: ({ column }) => <SortableHeader column={column} title="Gender" />,
      cell: ({ row }) => (
        <span className="text-muted-foreground">
          {row.original.gender === 'M' ? 'Male' : 'Female'}
        </span>
      ),
    },
    {
      accessorKey: 'age',
      header: ({ column }) => <SortableHeader column={column} title="Age" />,
      cell: ({ row }) => <span className="text-muted-foreground">{row.original.age}</span>,
    },
    {
      accessorKey: 'medical_record_number',
      header: ({ column }) => <SortableHeader column={column} title="MRN" />,
      cell: ({ row }) => (
        <span className="font-mono text-sm text-muted-foreground">
          {row.original.medical_record_number}
        </span>
      ),
    },
    {
      accessorKey: 'phone',
      header: 'Phone',
      cell: ({ row }) => (
        <span className="text-muted-foreground">
          {row.original.phone || '—'}
        </span>
      ),
    },
    {
      accessorKey: 'created_at',
      header: ({ column }) => <SortableHeader column={column} title="Registered" />,
      cell: ({ row }) => (
        <span className="text-muted-foreground">
          {new Date(row.original.created_at).toLocaleDateString()}
        </span>
      ),
      sortingFn: (rowA, rowB) => {
        return new Date(rowA.original.created_at).getTime() - new Date(rowB.original.created_at).getTime();
      },
    },
    {
      id: 'actions',
      header: '',
      cell: ({ row }) => (
        <ActionCell
          patient={row.original}
          onEdit={onEdit}
          onDelete={onDelete}
        />
      ),
      enableSorting: false,
    },
  ];
}
