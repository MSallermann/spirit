#pragma once
#ifndef HAMILTONIAN_MICROMAGNETIC_WIDGET_H
#define HAMILTONIAN_MICROMAGNETIC_WIDGET_H

#include <QtWidgets/QWidget>

#include <Spirit/Hamiltonian.h>

#include "SpinWidget.hpp"

#include "ui_HamiltonianMicromagneticWidget.h"

struct State;

class HamiltonianMicromagneticWidget : public QWidget, private Ui::HamiltonianMicromagneticWidget
{
    Q_OBJECT

public:
    HamiltonianMicromagneticWidget(std::shared_ptr<State> state, SpinWidget * spinWidget);
    void updateData();
    void setData();
signals:
    void hamiltonianChanged(Hamiltonian_Type newType);

private slots:
	void clicked_change_hamiltonian();
	void clicked_region();
    void set_boundary_conditions();
    void set_cell_sizes();
    void set_Ms();
    //void set_mu_s();
    void set_external_field();
    void set_anisotropy();

    void set_exchange();
    void set_dmi();
    void set_damping();
    void set_ddi_checkBox();
    void set_ddi_lineEdit();
    void set_frozen_spins();
    /*
    void set_ddi();
    void set_pairs_from_file();
    void set_pairs_from_text();*/

private:
    void Setup_Input_Validators();
    void Setup_Slots();
    std::shared_ptr<State> state;
    SpinWidget * spinWidget;

    // Validator for Input into lineEdits
    QRegularExpressionValidator * number_validator;
    QRegularExpressionValidator * number_validator_unsigned;
    QRegularExpressionValidator * number_validator_int;
    QRegularExpressionValidator * number_validator_int_unsigned;
};

#endif
